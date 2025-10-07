#!/usr/bin/env python3

import pandas as pd
import requests
import json
import os
import time
import getpass
from pathlib import Path

class ESPADownloader:
    """
    Class to handle ESPA API interactions for downloading Landsat data
    """
    def __init__(self, username: str, password: str, token: str, excel_file: str, output_dir: str):
        """
        Initialize the downloader with the Excel file path and output directory
        """
        self.excel_file = excel_file
        self.output_dir = output_dir
        self.host = "https://espa.cr.usgs.gov/api/v1/"
        # self.username = input("Enter ERS username: ")
        self.username = username
        self.password = password
        # self.password = getpass.getpass("Enter ERS password: ")
        self.token = token

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create a subdirectory for tar files
        self.tar_dir = os.path.join(output_dir, "tars")
        os.makedirs(self.tar_dir, exist_ok=True)

    def read_scene_ids(self):
        """
        Read scene IDs from the Excel file
        """
        df = pd.read_excel(self.excel_file)
        # Assuming the scene IDs are in column B labeled 'Scene ID'
        scene_ids = df['Scene ID'].tolist()
        return scene_ids

    def espa_api(self, endpoint, verb='get', body=None):
        """
        Helper function to interact with the ESPA API
        """
        # auth_tuple = (self.username, self.password)
        auth_tuple = (self.username, self.password)
        response = getattr(requests, verb)(self.host + endpoint, auth=auth_tuple, json=body)
        print(f'{response.status_code} {response.reason}')

        data = response.json()
        if isinstance(data, dict):
            messages = data.pop("messages", None)
            if messages:
                print(json.dumps(messages, indent=4))

        try:
            response.raise_for_status()
        except Exception as e:
            print(e)
            return None
        else:
            return data

    def check_available_products(self, scene_ids):
        """
        Check what products are available for the given scene IDs
        """
        avail_request = {'inputs': scene_ids}
        available = self.espa_api('available-products', body=avail_request)

        # Filter out any scene IDs that can't be processed
        valid_scenes = []
        for sensor in available.keys():
            if sensor != 'not_implemented' and isinstance(available[sensor], dict):
                valid_scenes.extend(available[sensor].get('inputs', []))

        return valid_scenes, available

    def place_order(self, scene_ids, available):
        """
        Place an order for the given scene IDs
        """
        # Define the order parameters
        order = available.copy()

        # Replace the products with what we want
        for sensor in order.keys():
            if isinstance(order[sensor], dict) and order[sensor].get('inputs'):
                # Customize products here based on what you want
                order[sensor]['products'] = ['l1', 'aq_refl']

        # Additional order parameters
        order['format'] = 'gtiff'
        order['resampling_method'] = 'cc'
        order['note'] = 'Order placed via Python ESPA API script'

        # Place the order
        print('Placing order...')
        response = self.espa_api('order', verb='post', body=order)

        if response:
            order_id = response.get('orderid')
            print(f'Order placed successfully with ID: {order_id}')
            return order_id
        else:
            print('Failed to place order')
            return None

    def check_order_status(self, order_id):
        """
        Check the status of an order
        """
        print(f'Checking status for order: {order_id}')
        status_data = self.espa_api(f'order-status/{order_id}')
        return status_data.get('status') if status_data else None

    def get_download_urls(self, order_id):
        """
        Get download URLs for completed items in an order
        """
        print(f'Getting download URLs for order: {order_id}')
        response = self.espa_api(f'item-status/{order_id}', body={'status': 'complete'})

        urls = []
        if response and order_id in response:
            for item in response[order_id]:
                if item.get('status') == 'complete' and item.get('product_dload_url'):
                    urls.append(item.get('product_dload_url'))

        return urls

    def download_file(self, url):
        """
        Download a file from the given URL
        """
        filename = os.path.basename(url)
        output_path = os.path.join(self.tar_dir, filename)

        if os.path.exists(output_path):
            print(f'File {filename} already exists, skipping download')
            return output_path

        print(f'Downloading {filename}...')
        start_time = time.time()

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            elapsed = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f'Downloaded {filename} ({file_size_mb:.2f} MB) in {elapsed:.2f} seconds')
            return output_path
        else:
            print(f'Failed to download {filename}')
            return None

    def extract_tar(self, tar_path):
        """
        Extract a tar.gz file to a subdirectory
        """
        import tarfile

        filename = os.path.basename(tar_path)
        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        extract_dir = os.path.join(self.output_dir, base_name)

        os.makedirs(extract_dir, exist_ok=True)

        print(f'Extracting {filename} to {extract_dir}...')
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        print(f'Extracted {filename} to {extract_dir}')
        return extract_dir

    def process(self):
        """
        Main processing function
        """
        # Read scene IDs from Excel
        scene_ids = self.read_scene_ids()
        print(f'Found {len(scene_ids)} scene IDs in Excel file')

        # Check available products
        valid_scenes, available = self.check_available_products(scene_ids)
        print(f'{len(valid_scenes)} valid scenes found')

        if not valid_scenes:
            print('No valid scenes found, exiting')
            return

        # Place order
        order_id = self.place_order(valid_scenes, available)
        if not order_id:
            return

        # Wait for order to complete
        print(f'Waiting for order {order_id} to complete...')
        status = self.check_order_status(order_id)

        while status == 'ordered':
            print(f'Order status: {status}. Waiting for 5 minutes before checking again...')
            time.sleep(300)  # Wait 5 minutes before checking again
            status = self.check_order_status(order_id)

        if status != 'complete':
            print(f'Order failed with status: {status}')
            return

        # Get download URLs
        print('Order complete. Getting download URLs...')
        urls = self.get_download_urls(order_id)
        print(f'Found {len(urls)} files to download')

        # Download files
        downloaded_files = []
        for url in urls:
            tar_path = self.download_file(url)
            if tar_path:
                downloaded_files.append(tar_path)

        # Extract tar files
        for tar_path in downloaded_files:
            self.extract_tar(tar_path)

        print('Processing complete!')

if __name__ == "__main__":
    # Get Excel file path and output directory from user
    # excel_file = input("Enter path to Excel file: ")
    # output_dir = input("Enter output directory path: ")

    excel_file = "/Volumes/Lacie/API/date_acquisitions_test.xlsx"
    output_dir = "/Volumes/Lacie/API/downloads"

    # Create and run the downloader
    downloader = ESPADownloader(excel_file, output_dir)
    downloader.process()
