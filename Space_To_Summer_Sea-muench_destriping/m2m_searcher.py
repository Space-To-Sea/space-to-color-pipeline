#!/usr/bin/env python3

import os
import yaml
import sys
import logging
import datetime
from pathlib import Path
import pandas as pd
import json
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("landsat_scene_finder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LandsatSceneFinder:
    def __init__(self, dates_list=False, dates_range=False, filters=None, params=None):
        """
        Initialize the Landsat scene finder

        Args:
            dates_list (bool): Use dates from Excel list
            dates_range (bool): Use date range for search
            filters (dict): Search filters
            params (dict): Additional parameters
        """
        self.dates_list = dates_list
        self.dates_range = dates_range
        self.filters = filters or {}
        self.params = params or {}

        # Extract credentials from params if available
        self.username = self.params.get('username')
        self.token = self.params.get('token')

        # API configuration
        self.serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
        self.dataset_name = "landsat_ot_c2_l1"

        # Default search parameters
        self.path = self.params.get('path', 11)
        self.row = self.params.get('row', 31)
        self.max_cloud_cover = self.filters.get('max_cloud_cover', 0.80)

        # Login and get API key
        self.apiKey = self.login()

        # Set up output directory based on search type
        if self.dates_list:
            self.excel_path = self.params.get('excel_path')
            self.output_excel_path = self.params.get('save_path')
            self.dates_column = self.params.get('dates_column', 'Date')
            self.already_ordered_column = self.params.get('already_ordered_column', 'Done By')
            self.filter_columns = self.params.get('filter_columns', {})
        elif self.dates_range:
            self.date_range = self.params.get('dates_range', [None, None])
            self.output_excel_path = self.params.get('save_path')
        else:
            raise ValueError("Either dates_list or dates_range must be True")

        # Create output directory if needed
        output_dir = os.path.dirname(self.output_excel_path)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results dataframe
        self.results_df = pd.DataFrame(columns=["Date", "Scene ID", "Cloud Cover"])

        logger.info(f"Initialized LandsatSceneFinder with output path: {self.output_excel_path}")

    def login(self):
        """
        Login to the USGS M2M API and get an API key

        Returns:
            str: API key for further requests
        """
        logger.info(f"Logging in with username: {self.username}")
        payload = {'username': self.username, 'token': self.token}

        apiKey = self.sendRequest(self.serviceUrl + "login-token", payload)
        logger.info("Successfully obtained API Key")

        return apiKey

    def logout(self):
        """
        Logout from the USGS M2M API
        """
        endpoint = "logout"
        result = self.sendRequest(self.serviceUrl + endpoint, None, self.apiKey)
        if result is None:
            logger.info("Logged Out")
        else:
            logger.error("Logout Failed")

    def read_excel(self):
        """
        Read the master Excel spreadsheet and find rows that need processing

        Returns:
            DataFrame: Filtered dataframe with rows that need processing
        """
        logger.info(f"Reading Excel file: {self.excel_path}")

        try:
            # Read Excel file with explicit date parsing
            df = pd.read_excel(self.excel_path)

            # Check if date column exists
            if self.dates_column not in df.columns:
                raise ValueError(f"The Excel file must contain a '{self.dates_column}' column.")

            # Try to parse dates
            df[self.dates_column] = pd.to_datetime(df[self.dates_column], errors='coerce')

            # Log column names for debugging
            logger.info(f"Excel columns: {df.columns.tolist()}")

            # Check for any invalid date formats
            date_count = df[self.dates_column].count()
            valid_date_count = df[self.dates_column].count()  # Already coerced to datetime
            if date_count != valid_date_count:
                logger.warning(f"Found {date_count - valid_date_count} rows with invalid date formats")

            # Filter for rows without a "Done By" value (or equivalent column) and that have valid dates
            to_process = df

            if self.already_ordered_column in df.columns:
                to_process = df[df[self.already_ordered_column].isna()]

            # Filter out rows with invalid dates
            to_process = to_process[~pd.isna(to_process[self.dates_column])]

            logger.info(f"Found {len(to_process)} rows to process")
            return to_process

        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return pd.DataFrame()

    def sendRequest(self, url, data, apiKey=None):
        """
        Send HTTP request to the USGS M2M API

        Args:
            url (str): API endpoint URL
            data (dict): Request payload
            apiKey (str, optional): API key for authentication

        Returns:
            dict/str: Response data
        """
        pos = url.rfind('/') + 1
        endpoint = url[pos:]

        # Convert data to JSON
        json_data = json.dumps(data) if data else None

        try:
            if apiKey is None:
                response = requests.post(url, data=json_data)
            else:
                headers = {'X-Auth-Token': apiKey}
                response = requests.post(url, data=json_data, headers=headers)

            # Check HTTP status code
            http_status_code = response.status_code
            if http_status_code != 200:
                logger.error(f"HTTP Error: {http_status_code}")
                response.close()
                return None

            # Parse response
            output = json.loads(response.text)

            # Check for API errors
            if output.get('errorCode'):
                logger.error(f"API Error: {output.get('errorCode')} - {output.get('errorMessage')}")
                logger.error(f"Request ID: {output.get('requestId')}")
                response.close()
                return None

            # Log success
            logger.info(f"Request {endpoint} completed with request ID {output.get('requestId', 'unknown')}")

            # Close response
            response.close()

            # Return data
            return output.get('data')

        except Exception as e:
            logger.error(f"Error in sendRequest for {endpoint}: {e}")
            if 'response' in locals() and response:
                response.close()
            return None

    def format_date_payload(self, date_obj):
        """
        Format the payload for scene ID request using a date

        Args:
            date_obj: A datetime object

        Returns:
            dict: Formatted payload for API request
        """
        logger.info(f"Formatting payload for date: {date_obj}")

        try:
            # Format as YYYY-MM-DD for API
            date_str = date_obj.strftime('%Y-%m-%d')

            # Create temporal filter
            temporal_filter = {'start': date_str, 'end': date_str}

            # Create metadata filter
            metadataFilter = {
                "filterType": "and",
                "childFilters": [
                    {'filterType': 'value', 'filterId': '5e83d14fb9436d88', 'value': self.path},  # WRS Path
                    {'filterType': 'value', 'filterId': '5e83d14ff1eda1b8', 'value': self.row},  # WRS Row
                    {'filterType': 'value', 'filterId': '5e83d14fc6e09eb6', 'value': 'OLI_TIRS'}  # Sensor Identifier
                ]
            }

            # Create payload structure
            payload = {
                'datasetName': self.dataset_name,
                'maxResults': 10,
                'sceneFilter': {
                    'acquisitionFilter': temporal_filter,
                    'metadataFilter': metadataFilter
                }
            }

            # Add cloud cover filter only if specified
            if self.max_cloud_cover is not None:
                payload['sceneFilter']['cloudCoverFilter'] = {
                    "max": self.max_cloud_cover,
                    "min": 0,
                    "includeUnknown": True
                }

            logger.info(f"Generated payload for date: {date_obj}")
            print(f"{payload=}")
            return payload

        except Exception as e:
            logger.error(f"Error formatting payload: {e}")
            return None

    def format_row_payload(self, row):
        """
        Format the payload for scene ID request using a row from dataframe

        Args:
            row: A row from the dataframe

        Returns:
            dict: Formatted payload for API request
        """
        logger.info(f"Formatting payload for row: {row.name}")

        # Extract date from row
        date_obj = row[self.dates_column]
        return self.format_date_payload(date_obj)

    def format_range_payload(self, start_date, end_date):
        """
        Format the payload for scene ID request using a date range

        Args:
            start_date: Start date string or datetime
            end_date: End date string or datetime

        Returns:
            dict: Formatted payload for API request
        """
        logger.info(f"Formatting payload for range: {start_date} to {end_date}")

        # Convert date strings to datetime objects if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Check if dates are valid
        if pd.isna(start_date) or pd.isna(end_date):
            logger.error(f"Invalid date range: {start_date} to {end_date}")
            return None

        try:
            # Format as YYYY-MM-DD for API
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Create temporal filter
            temporal_filter = {'start': start_date_str, 'end': end_date_str}

            # Create metadata filter
            metadataFilter = {
                "filterType": "and",
                "childFilters": [
                    {'filterType': 'value', 'filterId': '5e83d14fb9436d88', 'value': self.path},  # WRS Path
                    {'filterType': 'value', 'filterId': '5e83d14ff1eda1b8', 'value': self.row},  # WRS Row
                    {'filterType': 'value', 'filterId': '5e83d14fc6e09eb6', 'value': 'OLI_TIRS'}  # Sensor Identifier
                ]
            }

            # Create payload structure
            payload = {
                'datasetName': self.dataset_name,
                'maxResults': 100,  # Higher for date range queries
                'sceneFilter': {
                    'acquisitionFilter': temporal_filter,
                    'metadataFilter': metadataFilter
                }
            }

            # Add cloud cover filter only if specified
            if self.max_cloud_cover is not None:
                payload['sceneFilter']['cloudCoverFilter'] = {
                    "max": self.max_cloud_cover,
                    "min": 0,
                    "includeUnknown": True
                }

            logger.info(f"Generated payload for date range")
            return payload

        except Exception as e:
            logger.error(f"Error formatting range payload: {e}")
            return None

    def search_scenes(self, payload):
        """
        Search for scenes based on the provided payload

        Args:
            payload (dict): Search criteria payload

        Returns:
            list: List of scene metadata found
        """
        logger.info("Searching scenes...")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            url = self.serviceUrl + "scene-search"
            response = requests.post(
                url,
                json=payload,
                headers={'X-Auth-Token': self.apiKey}
            )

            logger.info(f"Response status: {response.status_code}")

            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response")
                return []

            # Check for API errors
            if response.status_code != 200 or response_data.get('errorCode'):
                error_msg = f"API Error: {response_data.get('errorCode')} - {response_data.get('errorMessage')}"
                logger.error(error_msg)
                return []

            # Get the scenes data
            scenes = response_data.get('data', {})

            # print(f"{scenes=}")

            if scenes.get('recordsReturned', 0) > 0:
                # Log the structure of the first result if available
                if scenes.get('results') and len(scenes.get('results')) > 0:
                    first_result = scenes['results'][0]
                    logger.info(f"First result keys: {list(first_result.keys())}")

                # Return the results
                return scenes.get('results', [])
            else:
                logger.info("Search found no results")
                return []

        except Exception as e:
            logger.error(f"Error searching scenes: {e}")
            return []

    def process_row(self, idx, row):
        """
        Process a single row from the dataframe

        Args:
            idx: Row index
            row: DataFrame row

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        logger.info(f"Processing row {idx}: {row[self.dates_column]}")

        # Format the payload
        payload = self.format_row_payload(row)
        if not payload:
            logger.error(f"Failed to create payload for row {idx}")
            return False

        # Search for scenes
        scenes = self.search_scenes(payload)

        # Check if any scenes were found
        if not scenes:
            logger.warning(f"No scenes found for row {idx}")
            return False

        # Process the scenes (add to results dataframe)
        for scene in scenes:
            # Extract scene ID and acquisition date
            # Check both displayId and entityId
            scene_id = scene.get('displayId')
            metadata = scene.get('metadata')
            date = None
            for dict in metadata:
                if 'ACQ_DATE' in dict.values():
                    date = dict['value']
            if date == None:
                logger.info(f"Found no acquisition date for {scene_id}")
                date = "Unknown"
            cloud_cover = scene.get('cloudCover')

            if scene_id:
                # Add to results dataframe
                self.results_df = pd.concat([
                    self.results_df,
                    pd.DataFrame({
                        "Date": [date],
                        "Scene ID": [scene_id],
                        "Cloud Cover": [cloud_cover]
                    })
                ], ignore_index=True)

                logger.info(f"Found scene ID {scene_id} for date {row[self.dates_column]}")

        return True

    def process_dates_from_excel(self):
        """
        Process dates from Excel file
        """
        if not self.dates_list:
            logger.error("Cannot process dates from Excel when dates_list is False")
            return

        # Read the Excel file
        df = self.read_excel()

        if df.empty:
            logger.info("No rows to process")
            return

        # Process each row
        success_count = 0
        for idx, row in df.iterrows():
            try:
                if self.process_row(idx, row):
                    success_count += 1
            except Exception as e:
                logger.error(f"Unhandled error processing row {idx}: {e}")
                continue

        # Write results to Excel file
        try:
            if not self.results_df.empty:
                # Write to Excel
                self.results_df.to_excel(self.output_excel_path, index=False)
                logger.info(f"Wrote {len(self.results_df)} scene IDs to {self.output_excel_path}")
            else:
                logger.warning("No scene IDs found to write to Excel")

                # Create an empty Excel file with headers
                empty_df = pd.DataFrame(columns=["Date", "Scene ID"])
                empty_df.to_excel(self.output_excel_path, index=False)
                logger.info(f"Created empty Excel file with headers at {self.output_excel_path}")
        except Exception as e:
            logger.error(f"Error writing to Excel: {e}")

        logger.info(f"Successfully processed {success_count} out of {len(df)} rows")

    def process_date_range(self):
        """
        Process date range
        """
        if not self.dates_range:
            logger.error("Cannot process date range when dates_range is False")
            return

        start_date, end_date = self.date_range

        # Check if dates are valid
        if not start_date or not end_date:
            logger.error(f"Invalid date range: {start_date} to {end_date}")
            return

        logger.info(f"Processing date range from {start_date} to {end_date}")

        # Format the payload
        payload = self.format_range_payload(start_date, end_date)
        if not payload:
            logger.error("Failed to create payload for date range")
            return

        # Search for scenes
        scenes = self.search_scenes(payload)

        # Check if any scenes were found
        if not scenes:
            logger.warning(f"No scenes found for date range")
            return

        # Process the scenes (add to results dataframe)
        for scene in scenes:
            # Extract scene ID and acquisition date - check both possible keys
            scene_id = scene.get('displayId') or scene.get('entityId')
            acquisition_date = scene.get('acquisitionDate') or scene.get('publishDate')

            if scene_id and acquisition_date:
                # Convert acquisition date to datetime
                try:
                    acq_date = pd.to_datetime(acquisition_date)

                    # Add to results dataframe
                    self.results_df = pd.concat([
                        self.results_df,
                        pd.DataFrame({
                            "Date": [acq_date],
                            "Scene ID": [scene_id]
                        })
                    ], ignore_index=True)

                    logger.info(f"Found scene ID {scene_id} for date {acq_date}")
                except Exception as e:
                    logger.warning(f"Could not parse acquisition date: {acquisition_date}, error: {e}")

        # Write results to Excel file
        try:
            if not self.results_df.empty:
                # Write to Excel
                self.results_df.to_excel(self.output_excel_path, index=False)
                logger.info(f"Wrote {len(self.results_df)} scene IDs to {self.output_excel_path}")
            else:
                logger.warning("No scene IDs found to write to Excel")

                # Create an empty Excel file with headers
                empty_df = pd.DataFrame(columns=["Date", "Scene ID"])
                empty_df.to_excel(self.output_excel_path, index=False)
                logger.info(f"Created empty Excel file with headers at {self.output_excel_path}")
        except Exception as e:
            logger.error(f"Error writing to Excel: {e}")

def main():
    """Main function to run the Landsat scene finder"""

    default_config_path = "/path/to/m2m_searcher.yaml"

    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else default_config_path
    except IndexError:
        config_path = default_config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    username = config['username']
    token = config['token']

    if config['dates_list']['use_dates_excel']:
        try:
            # Create scene finder for dates list
            finder = LandsatSceneFinder(
                dates_list=True,
                dates_range=False,
                filters=config['filters'],
                params={
                    'username': username,
                    'token': token,
                    'path': 11,
                    'row': 31,
                    'dates_column': config['dates_list']['dates_column'],
                    'already_ordered_column': config['dates_list']['already_ordered_column'],
                    'filter_columns': config['dates_list'].get('filter_columns', {}),
                    'excel_path': config['dates_list']['excel_path'],
                    'save_path': config['dates_list']['save_path']
                }
            )

            # Process all rows
            finder.process_dates_from_excel()

            # Logout
            finder.logout()

            logger.info("Script completed successfully")

        except Exception as e:
            logger.error(f"An error occurred in main: {e}")
            sys.exit(1)

    elif config['dates_range']['use_dates_range']:
        try:
            # Create scene finder for date range
            finder = LandsatSceneFinder(
                dates_list=False,
                dates_range=True,
                filters=config['filters'],
                params={
                    'username': username,
                    'token': token,
                    'path': 11,
                    'row': 31,
                    'dates_range': [
                        config['dates_range']['search_from']['start'],
                        config['dates_range']['search_from']['end']
                    ],
                    'save_path': config['dates_range']['save_path']
                }
            )

            # Process all rows
            finder.process_date_range()

            # Logout
            finder.logout()

            logger.info("Script completed successfully")

        except Exception as e:
            logger.error(f"An error occurred in main: {e}")
            sys.exit(1)

    else:
        logger.error("Please specify your choice of searching by dates or date range")
        sys.exit(1)

if __name__ == "__main__":
    main()
