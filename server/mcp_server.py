from fastmcp import FastMCP
from pydantic import BaseModel, Field
from agent.tools import filter_dataset, count_rows, get_unique_values, get_distribution, get_samples

mcp = FastMCP("Bitext_Data_Analyst")

class FilterDatasetInput(BaseModel):
    column_name: str = Field(description="The name of the column to filter on, strictly either 'intent' or 'category'")
    value: str = Field(description="The specific value to search for, e.g., 'get_refund', 'cancel_order', or 'ORDER'")

class GetDistributionInput(BaseModel):
    column_name: str = Field(description="The column to aggregate and count. Strictly 'category', 'intent', or 'flags'.")

class GetSamplesInput(BaseModel):
    category: str = Field(description="The specific category to retrieve text examples from, e.g., 'FEEDBACK', 'ORDER'")
    n: int = Field(description="The number of random examples to retrieve. Keep between 3 and 10.", default=5)


@mcp.tool()
def mcp_filter_dataset(args: FilterDatasetInput) -> str:
    """Filters the dataset down to rows that match a specific criteria."""

    return filter_dataset(args.column_name, args.value)

@mcp.tool()
def mcp_count_rows() -> str:
    """Counts the total number of rows in the currently active dataset."""

    return count_rows()

@mcp.tool()
def mcp_get_distribution(args: GetDistributionInput) -> str:
    """Returns a breakdown of all unique values and their exact counts for a specific column."""

    return get_distribution(args.column_name)

@mcp.tool()
def mcp_get_samples(args: GetSamplesInput) -> str:
    """Retrieves random sample user instructions from a specific category."""

    return get_samples(args.category, args.n)

@mcp.tool()
def mcp_get_unique_values() -> str:
    """Returns a list of all valid categories and intents. Use this if you are unsure what to filter by."""
    return get_unique_values()

if __name__ == "__main__":
    print("Starting FastMCP Server for Bitext Data Analyst...")
    mcp.run()