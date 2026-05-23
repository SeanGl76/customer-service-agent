import pandas as pd
from pathlib import Path

class DatasetManager:
    """
    A manager that holds the Pandas DataFrame in memory.
    It separates the base dataset from the filtered dataset
    so the agent can perform multi-step reasoning safely.
    """
    def __init__(self):
        current_dir = Path(__file__).parent
        csv_path = current_dir.parent / "data" / "bitext_dataset.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CRITICAL ERROR: Could not find the dataset at {csv_path}. "
                "Please ensure you downloaded 'bitext_dataset.csv' into the 'data' folder."
            )
        
        self.base_df = pd.read_csv(csv_path)
        
        if 'intent' in self.base_df.columns:
            self.base_df['intent'] = self.base_df['intent'].str.strip().str.lower()
        if 'category' in self.base_df.columns:
            self.base_df['category'] = self.base_df['category'].str.strip().str.upper()
            
        self.current_df = self.base_df.copy()

    def reset(self):
        """Wipes any active filters and restores the full dataset for the next question."""
        self.current_df = self.base_df.copy()

db = DatasetManager()

# STRUCTURED TOOLS

def filter_dataset(column_name: str, value: str) -> str:
    """Filters the dataset down to rows that match a specific criteria."""
    
    if column_name not in db.current_df.columns:
        return f"Tool Error: The column '{column_name}' does not exist in the dataset. Available columns are: {list(db.current_df.columns)}"
    
    if column_name == 'category':
        value = value.upper()
    elif column_name == 'intent':
        value = value.lower()
        
    db.current_df = db.current_df[db.current_df[column_name] == value]
    
    return f"Success. The dataset is now filtered by {column_name}='{value}'. The active dataset contains {len(db.current_df)} rows."


def count_rows() -> str:
    """Counts the total number of rows in the currently active dataset."""
    
    count = len(db.current_df)
    
    db.reset()
    
    return f"There are exactly {count} rows matching the current filters."


def get_unique_values() -> str:
    """Returns a list of all valid categories and intents. Use this if you are unsure what to filter by."""
    categories = db.base_df['category'].unique().tolist()
    intents = db.base_df['intent'].unique().tolist()
    
    return f"Valid Categories: {categories}\nValid Intents: {intents}"

def get_distribution(column_name: str) -> str:
    """Returns a breakdown of all unique values and their exact counts for a specific column. Use this for 'how many of each' or 'breakdown' questions."""
    
    if column_name not in db.current_df.columns:
        return f"Tool Error: The column '{column_name}' does not exist. Available columns are: {list(db.current_df.columns)}"
    
    distribution = db.current_df[column_name].value_counts().to_dict()
    
    db.reset()
    
    formatted_dist = "\n".join([f"- {key}: {count} occurrences" for key, count in distribution.items()])
    
    return f"Distribution of '{column_name}' in the current dataset:\n{formatted_dist}"


# 3. UNSTRUCTURED TOOLS (Qualitative Text)

def get_samples(category: str, n: int) -> str:
    """Retrieves random sample user instructions from a specific category. Use this to read text, find themes, or summarize."""
    
    # We use the base_df directly because qualitative extraction doesn't need to chain multiple tools together
    filtered_df = db.base_df[db.base_df['category'] == category.upper()]
    
    if filtered_df.empty:
        return f"Tool Error: No examples found for category '{category}'."
        
    # Safely get up to 'n' samples
    sample_df = filtered_df.sample(min(n, len(filtered_df)))
    
    # The Hugging Face dataset uses 'instruction' for the user's text
    instructions = sample_df['instruction'].tolist()
    
    # Format the output as a Markdown list so the LLM can parse it easily
    formatted_samples = "\n".join([f"- {text}" for text in instructions])
    
    return f"Here are {len(instructions)} user messages from the '{category}' category:\n{formatted_samples}"