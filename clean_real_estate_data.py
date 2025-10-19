import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)

class RealEstateDataCleaner:
    def __init__(self, input_file: str, output_dir: str = "cleaned_data"):
        """
        Initialize the RealEstateDataCleaner.
        
        Args:
            input_file: Path to the input CSV file
            output_dir: Directory to save cleaned data (default: 'cleaned_data')
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        
        # Property type mapping
        self.property_type_mapping = {
            'detached duplex': 'duplex',
            'semi-detached duplex': 'duplex',
            'terraced duplex': 'duplex',
            'flat / apartment': 'apartment',
            'flat': 'apartment',
            'apartment': 'apartment',
            'bungalow': 'bungalow',
            'detached bungalow': 'bungalow',
            'semi-detached bungalow': 'bungalow',
            'terraced bungalow': 'bungalow',
            'penthouse': 'penthouse',
            'maisonette': 'maisonette',
            'mansion': 'mansion',
            'duplex': 'duplex'
        }
        
        # Known LGAs in Nigeria (for location parsing)
        self.known_states = ['lagos', 'abuja', 'oyo', 'rivers', 'enugu', 'anambra', 'kaduna', 'kano', 'port harcourt', 'ibadan']
        self.known_lgas = {
            'lagos': ['lekki', 'ajah', 'victoria island', 'vi', 'ikeja', 'surulere', 'yaba', 'ikoyi', 'v.i', 'epe', 'ikorodu'],
            'abuja': ['asokoro', 'maitama', 'wuse', 'gwarinpa', 'jabi', 'garki', 'wuse 2', 'wuse ii', 'central business district', 'cbd'],
            'oyo': ['ibadan north', 'ibadan south', 'ibadan northeast', 'ibadan northwest', 'ibadan southeast', 'ibadan southwest']
        }
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the input file, automatically detecting CSV or TSV format.
        
        Returns:
            pd.DataFrame: The loaded DataFrame
        """
        try:
            # First try reading as CSV with comma delimiter
            try:
                self.df = pd.read_csv(
                    self.input_file,
                    encoding='utf-8',
                    on_bad_lines='warn',
                    dtype=str
                )
                # If we only have 1 column, it's probably a TSV file
                if len(self.df.columns) == 1:
                    self.df = pd.read_csv(
                        self.input_file,
                        sep='\t',
                        encoding='utf-8',
                        on_bad_lines='warn',
                        dtype=str
                    )
            except Exception as e:
                # If CSV reading fails, try as TSV
                self.df = pd.read_csv(
                    self.input_file,
                    sep='\t',
                    encoding='utf-8',
                    on_bad_lines='warn',
                    dtype=str
                )
            
            # Clean column names
            self.clean_column_names()
            
            # Convert empty strings to NaN
            self.df = self.df.replace(r'^\s*$', pd.NA, regex=True)
            
            logging.info(f"Loaded {len(self.df)} records from {self.input_file}")
            logging.info(f"Columns detected: {', '.join(self.df.columns)}")
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def clean_column_names(self) -> None:
        """Standardize column names (lowercase, underscores)."""
        self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]
        logging.info("Standardized column names")
    
    def remove_duplicates(self) -> None:
        """Remove duplicate listings based on title + location + price."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['title', 'location', 'price'])
        removed = initial_count - len(self.df)
        logging.info(f"Removed {removed} duplicate listings")
    
    def clean_price(self) -> None:
        """Convert price to numeric by removing currency symbols and commas."""
        if 'price' in self.df.columns:
            # Remove currency symbols and commas, then convert to numeric
            self.df['price_numeric'] = (
                self.df['price']
                .astype(str)
                .str.replace(r'[^\d.]', '', regex=True)
                .replace('', np.nan)
                .astype(float)
            )
            logging.info("Converted price to numeric format")
    
    def handle_missing_values(self) -> None:
        """Fill missing values using median values grouped by property type and location."""
        numeric_cols = ['bedrooms', 'bathrooms', 'toilets']
        
        for col in numeric_cols:
            if col in self.df.columns:
                # Convert to string first to handle mixed types, then to numeric
                self.df[col] = pd.to_numeric(self.df[col].astype(str).str.extract('(\d+)')[0], errors='coerce')
                
                # Calculate median by property type and location
                if 'property_type' in self.df.columns and 'location' in self.df.columns:
                    self.df[col] = self.df[col].fillna(
                        self.df.groupby(['property_type', 'location'])[col].transform('median')
                    )
                
                # Fill any remaining NaNs with overall median
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
                # Convert to integer (since you can't have a fraction of a room)
                self.df[col] = self.df[col].round().astype('Int64')
                
                logging.info(f"Handled missing values for {col}")
    
    def normalize_property_type(self) -> None:
        """Standardize property type values."""
        if 'property_type' in self.df.columns:
            # Clean and standardize property type strings
            self.df['property_type'] = self.df['property_type'].str.lower().str.strip()
            
            # Map to standardized values
            self.df['property_type_standardized'] = (
                self.df['property_type']
                .map(lambda x: next((v for k, v in self.property_type_mapping.items() 
                                   if k in x), 'other'))
            )
            logging.info("Standardized property types")
    
    def parse_location(self, location: str) -> Tuple[str, str, str]:
        """Parse location string into state, LGA, and district/estate."""
        if not isinstance(location, str):
            return None, None, None
            
        parts = [p.strip() for p in location.lower().split(',')]
        parts = [p for p in parts if p]  # Remove empty strings
        
        if not parts:
            return None, None, None
        
        # Initialize variables
        state = None
        lga = None
        district = None
        
        # Try to find state (usually the last part)
        for i in range(len(parts)-1, -1, -1):
            part = parts[i].lower()
            # Check if part matches any known state
            for known_state in self.known_states:
                if known_state in part:
                    state = known_state.title()
                    parts.pop(i)
                    break
            if state:
                break
        
        # If no state found, try to infer from LGAs
        if not state:
            for part in parts:
                for known_state, lgas in self.known_lgas.items():
                    if any(lga in part for lga in lgas):
                        state = known_state.title()
                        break
                if state:
                    break
        
        # Try to find LGA (usually the middle part)
        if len(parts) > 1:
            # Check if any part matches known LGAs
            for i, part in enumerate(parts):
                part_lower = part.lower()
                if state and state.lower() in self.known_lgas:
                    for lga_name in self.known_lgas[state.lower()]:
                        if lga_name in part_lower:
                            lga = part.title()
                            parts.pop(i)
                            break
                if lga:
                    break
            
            # If no LGA found, use the last part before state (or last part if no state)
            if not lga and parts:
                lga = parts[-1].title()
                parts = parts[:-1]
        
        # The remaining parts are considered the district/estate
        if parts:
            district = ', '.join(parts).title()
        
        return state, lga, district
    
    def normalize_location(self) -> None:
        """Parse and split location into structured components."""
        if 'location' in self.df.columns:
            # Apply the parse_location function to each location
            location_components = self.df['location'].apply(self.parse_location)
            
            # Create new columns
            self.df[['state', 'lga', 'district_or_estate']] = pd.DataFrame(
                location_components.tolist(), 
                index=self.df.index
            )
            
            # Clean up the location strings
            for col in ['state', 'lga', 'district_or_estate']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].str.title()
            
            logging.info("Parsed location information")
    
    def standardize_text(self) -> None:
        """Standardize text fields by trimming whitespace and normalizing case."""
        text_columns = ['title', 'location', 'property_type']
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
        
        logging.info("Standardized text fields")
    
    def validate_numeric_ranges(self) -> None:
        """Ensure numeric fields are within realistic bounds."""
        numeric_ranges = {
            'price_numeric': {'min': 10000, 'max': 5000000000},
            'bedrooms': {'min': 0, 'max': 20},
            'bathrooms': {'min': 0, 'max': 20},
            'toilets': {'min': 0, 'max': 20}
        }
        
        for col, bounds in numeric_ranges.items():
            if col in self.df.columns:
                # Convert to numeric if not already
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                # Apply bounds
                self.df[col] = self.df[col].clip(
                    lower=bounds['min'],
                    upper=bounds['max']
                )
                
                logging.info(f"Validated ranges for {col}")
    
    def save_cleaned_data(self, output_file: str = None) -> str:
        """
        Save the cleaned data to a CSV file.
        
        Args:
            output_file: Output file path (default: 'cleaned_<input_filename>')
            
        Returns:
            str: Path to the saved file
        """
        if output_file is None:
            input_stem = Path(self.input_file).stem
            output_file = self.output_dir / f"cleaned_{input_stem}.csv"
        else:
            output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Select and reorder columns for output
        output_columns = [
            'title',
            'price_numeric',
            'state',
            'lga',
            'district_or_estate',
            'bedrooms',
            'bathrooms',
            'toilets',
            'property_type_standardized',
            'location',  # Keep original location for reference
            'property_type'  # Keep original property type for reference
        ]
        
        # Only include columns that exist in the DataFrame
        output_columns = [col for col in output_columns if col in self.df.columns]
        
        # Save to CSV
        self.df[output_columns].to_csv(output_file, index=False)
        logging.info(f"Saved cleaned data to {output_file}")
        
        return str(output_file)
    
    def run_cleaning_pipeline(self) -> str:
        """Run the complete data cleaning pipeline."""
        logging.info("Starting data cleaning pipeline")
        
        try:
            # Load the data (clean_column_names is called within load_data)
            self.load_data()
            
            # Standardize text fields
            self.standardize_text()
            
            # Handle duplicates
            self.remove_duplicates()
            
            # Clean and convert price
            self.clean_price()
            
            # Normalize property types
            self.normalize_property_type()
            
            # Parse location information
            self.normalize_location()
            
            # Handle missing values (after location parsing)
            self.handle_missing_values()
            
            # Validate numeric ranges
            self.validate_numeric_ranges()
            
            # Save cleaned data
            output_file = self.save_cleaned_data()
            
            logging.info("Data cleaning completed successfully")
            return output_file
            
        except Exception as e:
            logging.error(f"Error in cleaning pipeline: {e}")
            raise


def main():
    """Main function to run the data cleaning."""
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Clean and normalize real estate data')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to save the cleaned output file (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the cleaning pipeline
    try:
        cleaner = RealEstateDataCleaner(args.input_file)
        output_file = cleaner.run_cleaning_pipeline()
        print(f"\nCleaning complete! Output saved to: {output_file}")
    except Exception as e:
        print(f"\nError during cleaning: {e}")
        raise


if __name__ == "__main__":
    main()
