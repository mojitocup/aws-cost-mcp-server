"""
AWS Cost Explorer MCP Server.

This server provides MCP tools to interact with AWS Cost Explorer API.
"""
import os
import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Literal

import boto3
import pandas as pd
import json
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from tabulate import tabulate



class DaysParam(BaseModel):
    """Parameters for specifying the number of days to look back."""
    
    days: int = Field(
        default=7,
        description="Number of days to look back for cost data"
    )



class BedrockLogsParams(BaseModel):
    """Parameters for retrieving Bedrock invocation logs."""
    days: int = Field(
        default=7,
        description="Number of days to look back for Bedrock logs",
        ge=1,
        le=90
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region to retrieve logs from"
    )
    log_group_name: str = Field(
        description="Bedrock Log Group Name",
        default=os.environ.get('BEDROCK_LOG_GROUP_NAME', 'BedrockModelInvocationLogGroup')
    )
    aws_account_id: Optional[str] = Field(        
        description="AWS account id (if different from the current AWS account) of the account for which to get the cost data",
        default=None
    )

class EC2Params(BaseModel):
    """Parameters for retrieving EC2 Cost Explorer information."""
    days: int = Field(
        default=1,
        description="Number of days to look back for Bedrock logs",
        ge=1,
        le=90
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region to retrieve logs from"
    )
    aws_account_id: Optional[str] = Field(        
        description="AWS account id (if different from the current AWS account) of the account for which to get the cost data",
        default=None
    )
    ec2_record_type_filter: List[str] = Field(
        default_factory=lambda: ["Usage"],
        description=(
            "Cost Explorer RECORD_TYPE values included for EC2 spend-by-instance-type and nested EC2 instance breakdowns. "
            "Default ['Usage'] matches Console-style on-demand usage rows and excludes RIFee / DiscountedUsage amortization "
            "that can spike UnblendedCost when grouped by INSTANCE_TYPE. Pass an empty list to disable (legacy all record types)."
        ),
    )


class CostExplorerQueryParams(BaseModel):
    """Parameters for generic Cost Explorer cost and usage queries."""
    days: int = Field(
        default=30,
        description="Number of days to look back for cost data",
        ge=1,
        le=365
    )
    granularity: Literal["DAILY", "MONTHLY"] = Field(
        default="DAILY",
        description="Granularity of the returned cost data"
    )
    metrics: List[Literal["UnblendedCost", "BlendedCost", "AmortizedCost", "NetAmortizedCost", "NetUnblendedCost", "UsageQuantity", "NormalizedUsageAmount"]] = Field(
        default=["UnblendedCost"],
        description="Metrics to retrieve from Cost Explorer"
    )
    group_by_dimension: Optional[Literal["SERVICE", "REGION", "LINKED_ACCOUNT", "INSTANCE_TYPE", "USAGE_TYPE", "RECORD_TYPE"]] = Field(
        default=None,
        description="Optional Cost Explorer dimension to group results by"
    )
    service_filter: Optional[str] = Field(
        default=None,
        description="Optional service filter, e.g. 'Amazon Elastic Compute Cloud - Compute'"
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region to retrieve billing data from"
    )
    aws_account_id: Optional[str] = Field(
        description="AWS account id (if different from the current AWS account) of the account for which to get the cost data",
        default=None
    )
    record_type_filter: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional Cost Explorer RECORD_TYPE filter (AND with service_filter if set). "
            "Use e.g. ['Usage'] to align instance-type costs with on-demand-style rows; omit for all record types."
        ),
    )


class ReservationParams(BaseModel):
    """Parameters for reservation utilization and coverage queries."""
    days: int = Field(
        default=30,
        description="Number of days to look back for reservation data",
        ge=1,
        le=365
    )
    granularity: Literal["DAILY", "MONTHLY"] = Field(
        default="DAILY",
        description="Granularity for reservation data"
    )
    region: str = Field(
        default="us-east-1",
        description="AWS region to retrieve billing data from"
    )
    aws_account_id: Optional[str] = Field(
        description="AWS account id (if different from the current AWS account) of the account for which to get the cost data",
        default=None
    )


class ReservationSummaryParams(ReservationParams):
    """Parameters for human-friendly reservation summary reports."""
    service: Literal["EC2", "RDS", "BOTH"] = Field(
        default="BOTH",
        description="Reservation service to summarize"
    )


class EC2RIFlexibilityReportParams(BaseModel):
    """Parameters for EC2 regional RI instance-size flexibility vs running footprint."""

    region: str = Field(
        default="us-east-1",
        description="AWS region to query EC2 APIs in",
    )
    aws_account_id: Optional[str] = Field(
        default=None,
        description="AWS account id (if different from the current AWS account)",
    )
    platform_scope: Literal["linux_unix", "windows", "all"] = Field(
        default="linux_unix",
        description="Which OS family to include when matching RIs to running instances (instance size flexibility applies per AWS rules; linux_unix is the common regional flexible case).",
    )


class ComputeOptimizerEC2Params(BaseModel):
    """Parameters for EC2 Compute Optimizer recommendations."""
    region: str = Field(
        default="us-east-1",
        description="AWS region to retrieve Compute Optimizer findings from"
    )
    aws_account_id: Optional[str] = Field(
        description="AWS account id (if different from the current AWS account)",
        default=None
    )
    finding: Optional[Literal["Underprovisioned", "Overprovisioned", "Optimized", "NotOptimized"]] = Field(
        default=None,
        description="Optional finding filter"
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of recommendations to return"
    )


class ComputeOptimizerRDSParams(BaseModel):
    """Parameters for RDS Compute Optimizer recommendations."""
    region: str = Field(
        default="us-east-1",
        description="AWS region to retrieve Compute Optimizer findings from"
    )
    aws_account_id: Optional[str] = Field(
        description="AWS account id (if different from the current AWS account)",
        default=None
    )
    finding: Optional[Literal["Underprovisioned", "Overprovisioned", "Optimized", "NotOptimized"]] = Field(
        default=None,
        description="Optional finding filter"
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of recommendations to return"
    )


class RightsizingRecommendationParams(BaseModel):
    """Parameters for Cost Explorer rightsizing recommendations."""
    region: str = Field(
        default="us-east-1",
        description="AWS region for Cost Explorer API"
    )
    aws_account_id: Optional[str] = Field(
        description="AWS account id (if different from the current AWS account)",
        default=None
    )
    service: Literal["AmazonEC2"] = Field(
        default="AmazonEC2",
        description="Service for rightsizing recommendation"
    )
    lookback_period: Literal["SEVEN_DAYS", "THIRTY_DAYS", "SIXTY_DAYS"] = Field(
        default="THIRTY_DAYS",
        description="Lookback period for rightsizing recommendation"
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Page size for rightsizing recommendation results"
    )


class SavingsPlansRecommendationParams(BaseModel):
    """Parameters for Cost Explorer Savings Plans purchase recommendations."""
    region: str = Field(
        default="us-east-1",
        description="AWS region for Cost Explorer API"
    )
    aws_account_id: Optional[str] = Field(
        description="AWS account id (if different from the current AWS account)",
        default=None
    )
    lookback_period: Literal["SEVEN_DAYS", "THIRTY_DAYS", "SIXTY_DAYS"] = Field(
        default="THIRTY_DAYS",
        description="Lookback period for Savings Plans recommendation"
    )
    payment_option: Literal["NO_UPFRONT", "PARTIAL_UPFRONT", "ALL_UPFRONT"] = Field(
        default="NO_UPFRONT",
        description="Savings Plans payment option"
    )
    term_in_years: Literal["ONE_YEAR", "THREE_YEARS"] = Field(
        default="ONE_YEAR",
        description="Savings Plans term"
    )
# global params
# if we want to get AWS spend info from a different account we need to assume a role in that account
# and while the account id would be provided by the user of this MCP server, we set the name of the role
# to assume in this code through an environ variable
CROSS_ACCOUNT_ROLE_NAME: str = os.environ.get('CROSS_ACCOUNT_ROLE_NAME', "BedrockCrossAccount2")

def get_aws_service_boto3_client(service: str, aws_account_id: Optional[str], region_name: str, account_b_role_name: Optional[str] = CROSS_ACCOUNT_ROLE_NAME):
    """
    Creates a boto3 client for the specified service in this current AWS account or in a different account
    if an account id is specified.
    
    Args:
        service (str): AWS service name (e.g., 'logs', 'cloudwatch')
        region_name (str): AWS region (e.g. 'us-east-1')
        aws_account_id (str): AWS account ID to access, this is the account in which the role is to be assumed
        account_b_role_name (str): IAM role name to assume
        
    Returns:
        boto3.client: Service client with assumed role credentials
    """
    try:
        this_account = boto3.client('sts').get_caller_identity()['Account']
        if aws_account_id is not None and this_account != aws_account_id:
            # the request is for a different account, we need to assume a role in that account
            print(f"Request is for a different account: {aws_account_id}, current account: {this_account}")
            # Create STS client
            sts_client = boto3.client('sts')
            current_identity = sts_client.get_caller_identity()
            print(f"Current identity: {current_identity}")
            
            # Define the role ARN
            role_arn = f"arn:aws:iam::{aws_account_id}:role/{account_b_role_name}"
            print(f"Attempting to assume role: {role_arn}")
            
            # Assume the role
            assumed_role = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="CrossAccountSession"
            )
            
            # Extract temporary credentials
            credentials = assumed_role['Credentials']
            
            # Create client with assumed role credentials
            client = boto3.client(
                service,
                region_name=region_name,
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            
            print(f"Successfully created cross-account client for {service} in account {aws_account_id}")
            return client
        else:
            client = boto3.client(
                service,
                region_name=region_name
            )
            
            print(f"Successfully created client for {service} in the current AWS account {this_account}")
            return client
        
    except Exception as e:
        print(f"Error creating cross-account client for {service}: {e}")
        raise e


def _cost_explorer_dimension_filter(
    *,
    service_values: Optional[List[str]] = None,
    record_type_values: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a Cost Explorer Filter combining SERVICE and/or RECORD_TYPE dimensions.

    When both are present, wraps them in an And expression (required by the API).
    """
    clauses: List[Dict[str, Any]] = []
    if service_values:
        clauses.append({"Dimensions": {"Key": "SERVICE", "Values": service_values}})
    if record_type_values:
        clauses.append({"Dimensions": {"Key": "RECORD_TYPE", "Values": record_type_values}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"And": clauses}


def get_bedrock_logs(params: BedrockLogsParams) -> Optional[pd.DataFrame]:
    """
    Retrieve Bedrock invocation logs for the last n days in a given region as a dataframe

    Args:
        params: Pydantic model containing parameters:
            - days: Number of days to look back (default: 7)
            - region: AWS region to query (default: us-east-1)

    Returns:
        pd.DataFrame: DataFrame containing the log data with columns:
            - timestamp: Timestamp of the invocation
            - region: AWS region
            - modelId: Bedrock model ID
            - userId: User ARN
            - inputTokens: Number of input tokens
            - completionTokens: Number of completion tokens
            - totalTokens: Total tokens used
    """
    # Initialize CloudWatch Logs client
    print(f"get_bedrock_logs, params={params}")
    client = get_aws_service_boto3_client("logs", params.aws_account_id, params.region)

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=params.days)

    # Convert to milliseconds since epoch
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    filtered_logs = []

    try:
        paginator = client.get_paginator("filter_log_events")

        # Parameters for the log query        
        query_params = {
            "logGroupName": params.log_group_name,  # Use the provided log group name
            "logStreamNames": [
                "aws/bedrock/modelinvocations"
            ],  # The specific log stream
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        }
        
        # Paginate through results
        for page in paginator.paginate(**query_params):
            for event in page.get("events", []):
                try:
                    # Parse the message as JSON

                    message = json.loads(event["message"])

                    # Get user prompt from the input messages
                    prompt = ""
      
                    input = message.get("input", {})
                    input_json = input.get("inputBodyJson", {})
                    messages = input_json.get("messages", None)

                    if messages:
                        for msg in message["input"]["inputBodyJson"]["messages"]:
                            #print(f"debug 2.2, {type(msg)}")
                            if msg.get("role") == "user" and msg.get("content"):
                                for content in msg["content"]:

                                    if isinstance(content, dict):
                                        if content.get("text"):
                                            prompt += content["text"] + " "
                                    else:
                                        prompt += content

                        prompt = prompt.strip()

                    # Extract only the required fields

                    filtered_event = {
                        "timestamp": message.get("timestamp"),
                        "region": message.get("region"),
                        "modelId": message.get("modelId"),
                        "userId": message.get("identity", {}).get("arn"),
                        "inputTokens": message.get("input", {}).get("inputTokenCount"),
                        "completionTokens": message.get("output", {}).get(
                            "outputTokenCount"
                        ),
                        "totalTokens": (
                            message.get("input", {}).get("inputTokenCount", 0)
                            + message.get("output", {}).get("outputTokenCount", 0)
                        ),
                    }

                    filtered_logs.append(filtered_event)
                except json.JSONDecodeError:
                    continue  # Skip non-JSON messages
                except KeyError:
                    continue  # Skip messages missing required fields
        
        # Create DataFrame if we have logs
        if filtered_logs:
            df = pd.DataFrame(filtered_logs)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        else:
            print("No logs found for the specified time period.")
            return None

    except client.exceptions.ResourceNotFoundException:
        print(
            f"Log group '{params.log_group_name}' or stream 'aws/bedrock/modelinvocations' not found"
        )
        return None
    except Exception as e:
        print(f"Error retrieving logs: {str(e)}")
        return None



# Initialize FastMCP server
mcp = FastMCP("aws_cloudwatch_logs")
@mcp.prompt()
def system_prompt_for_agent(aws_account_id: str = "") -> str:
    """
    Generates a system prompt for an AWS cost analysis agent.
    
    This function creates a specialized prompt for an AI agent that analyzes
    AWS cloud spending. The prompt instructs the agent on how to retrieve,
    analyze, and present cost optimization insights for AWS accounts.
    
    Args:
        aws_account_id (Optional[str]): The AWS account ID to analyze.
            If provided, the agent will focus on this specific account.
            If None, the agent will function without account-specific context.
    
    Returns:
        str: A formatted system prompt for the AWS cost analysis agent.
    """
    if aws_account_id == "":
        aws_account_id = boto3.client('sts').get_caller_identity()['Account']
    account_context = f"for account {aws_account_id}"
    initial_line = f"You are an expert AWS cost analyst AI agent {account_context}."
    second_line = f"Your purpose is to help users understand and optimize their AWS cloud spending for this account."
    
    system_prompt = f"""
{initial_line} {second_line} You have access to the following tools:

1. AWS Cost Explorer data retrieval
2. CloudWatch logs analysis
3. Resource tagging information
4. Billing data by account, service, and region
5. Historical spend pattern analysis
6. AWS-native optimization recommendations (Cost Explorer and Compute Optimizer findings)
7. EC2 regional Reserved Instance capacity vs running footprint (AWS normalization units for instance size flexibility)

When a user asks about their AWS costs:

1. First, retrieve relevant data using your tools
2. Analyze spending patterns across services, users, applications, and time periods
3. Identify:
   - Highest cost services and resources
   - Unused or underutilized resources
   - Spending anomalies and unexpected increases
   - Resources lacking proper cost allocation tags
   - Opportunities for reserved instances or savings plans
   - Potential architectural optimizations

4. Present findings in a clear, actionable format with:
   - Visual breakdowns of cost distribution
   - Specific recommendations for cost optimization
   - Estimated potential savings for each recommendation
   - Comparative analysis with previous time periods

Respond to queries about specific services, accounts, or time periods with precise, data-backed insights. Always provide practical recommendations that balance cost optimization with operational requirements.
"""
    return system_prompt

@mcp.tool()
def get_bedrock_daily_usage_stats(params: BedrockLogsParams) -> str:
    """
    Get daily usage statistics with detailed breakdowns.

    Args:
        params: Parameters specifying the number of days to look back and region

    Returns:
        str: Formatted string representation of daily usage statistics
    """
    print(f"get_bedrock_daily_usage_stats, params={params}")
    df = get_bedrock_logs(params)

    if df is None or df.empty:
        return "No usage data found for the specified period."
    
    # Initialize result string
    result_parts = []
    
    # Add header
    result_parts.append(f"Bedrock Usage Statistics (Past {params.days} days - {params.region})")
    result_parts.append("=" * 80)
    
    # Add a date column for easier grouping
    df['date'] = df['timestamp'].dt.date
    
    # === REGION -> MODEL GROUPING ===
    result_parts.append("\n=== Daily Region-wise -> Model-wise Analysis ===")
    
    # Group by date, region, model and calculate metrics
    region_model_stats = df.groupby(['date', 'region', 'modelId']).agg({
        'inputTokens': ['count', 'sum', 'mean', 'max', 'median'],
        'completionTokens': ['sum', 'mean', 'max', 'median'],
        'totalTokens': ['sum', 'mean', 'max', 'median']
    })
    
    # Flatten the column multi-index
    region_model_stats.columns = [f"{col[0]}_{col[1]}" for col in region_model_stats.columns]
    
    # Reset the index to get a flat dataframe
    flattened_stats = region_model_stats.reset_index()
    
    # Rename inputTokens_count to request_count
    flattened_stats = flattened_stats.rename(columns={'inputTokens_count': 'request_count'})
    
    # Add the flattened stats to result
    result_parts.append(flattened_stats.to_string(index=False))
    
    # Add summary statistics
    result_parts.append("\n=== Summary Statistics ===")
    
    # Total requests and tokens
    total_requests = flattened_stats['request_count'].sum()
    total_input_tokens = flattened_stats['inputTokens_sum'].sum()
    total_completion_tokens = flattened_stats['completionTokens_sum'].sum()
    total_tokens = flattened_stats['totalTokens_sum'].sum()
    
    result_parts.append(f"Total Requests: {total_requests:,}")
    result_parts.append(f"Total Input Tokens: {total_input_tokens:,}")
    result_parts.append(f"Total Completion Tokens: {total_completion_tokens:,}")
    result_parts.append(f"Total Tokens: {total_tokens:,}")
    
    # === REGION SUMMARY ===
    result_parts.append("\n=== Region Summary ===")
    region_summary = df.groupby('region').agg({
        'inputTokens': ['count', 'sum'],
        'completionTokens': ['sum'],
        'totalTokens': ['sum']
    })
    
    # Flatten region summary columns
    region_summary.columns = [f"{col[0]}_{col[1]}" for col in region_summary.columns]
    region_summary = region_summary.reset_index()
    region_summary = region_summary.rename(columns={'inputTokens_count': 'request_count'})
    
    result_parts.append(region_summary.to_string(index=False))
    
    # === MODEL SUMMARY ===
    result_parts.append("\n=== Model Summary ===")
    model_summary = df.groupby('modelId').agg({
        'inputTokens': ['count', 'sum'],
        'completionTokens': ['sum'],
        'totalTokens': ['sum']
    })
    
    # Flatten model summary columns
    model_summary.columns = [f"{col[0]}_{col[1]}" for col in model_summary.columns]
    model_summary = model_summary.reset_index()
    model_summary = model_summary.rename(columns={'inputTokens_count': 'request_count'})
    
    # Format model IDs to be more readable
    model_summary['modelId'] = model_summary['modelId'].apply(
        lambda model: model.split('.')[-1] if '.' in model else model.split('/')[-1]
    )
    
    result_parts.append(model_summary.to_string(index=False))
    
    # === USER SUMMARY ===
    if 'userId' in df.columns:
        result_parts.append("\n=== User Summary ===")
        user_summary = df.groupby('userId').agg({
            'inputTokens': ['count', 'sum'],
            'completionTokens': ['sum'],
            'totalTokens': ['sum']
        })
        
        # Flatten user summary columns
        user_summary.columns = [f"{col[0]}_{col[1]}" for col in user_summary.columns]
        user_summary = user_summary.reset_index()
        user_summary = user_summary.rename(columns={'inputTokens_count': 'request_count'})
        
        result_parts.append(user_summary.to_string(index=False))
        
    # === REGION -> USER -> MODEL DETAILED SUMMARY ===
    if 'userId' in df.columns:
        result_parts.append("\n=== Region -> User -> Model Detailed Summary ===")
        region_user_model_summary = df.groupby(['region', 'userId', 'modelId']).agg({
            'inputTokens': ['count', 'sum', 'mean'],
            'completionTokens': ['sum', 'mean'],
            'totalTokens': ['sum', 'mean']
        })
        
        # Flatten columns
        region_user_model_summary.columns = [f"{col[0]}_{col[1]}" for col in region_user_model_summary.columns]
        region_user_model_summary = region_user_model_summary.reset_index()
        region_user_model_summary = region_user_model_summary.rename(columns={'inputTokens_count': 'request_count'})
        
        # Format model IDs to be more readable
        region_user_model_summary['modelId'] = region_user_model_summary['modelId'].apply(
            lambda model: model.split('.')[-1] if '.' in model else model.split('/')[-1]
        )
        
        result_parts.append(region_user_model_summary.to_string(index=False))

    
    # Combine all parts into a single string
    result = "\n".join(result_parts)
    
    return result

@mcp.tool()
def get_bedrock_hourly_usage_stats(params: BedrockLogsParams) -> str:
    """
    Get hourly usage statistics with detailed breakdowns.

    Args:
        params: Parameters specifying the number of days to look back and region

    Returns:
        str: Formatted string representation of hourly usage statistics
    """
    print(f"get_bedrock_hourly_usage_stats, params={params}")
    df = get_bedrock_logs(params)

    if df is None or df.empty:
        return "No usage data found for the specified period."
    
    # Initialize result string
    result_parts = []
    
    # Add header
    result_parts.append(f"Hourly Bedrock Usage Statistics (Past {params.days} days - {params.region})")
    result_parts.append("=" * 80)
    
    # Add date and hour columns for easier grouping
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['datetime'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00')
    
    # === HOURLY USAGE ANALYSIS ===
    result_parts.append("\n=== Hourly Usage Analysis ===")
    
    # Group by datetime (date + hour)
    hourly_stats = df.groupby('datetime').agg({
        'inputTokens': ['count', 'sum', 'mean'],
        'completionTokens': ['sum', 'mean'],
        'totalTokens': ['sum', 'mean']
    })
    
    # Flatten the column multi-index
    hourly_stats.columns = [f"{col[0]}_{col[1]}" for col in hourly_stats.columns]
    
    # Reset the index to get a flat dataframe
    hourly_stats = hourly_stats.reset_index()
    
    # Rename inputTokens_count to request_count
    hourly_stats = hourly_stats.rename(columns={'inputTokens_count': 'request_count'})
    
    # Add the hourly stats to result
    result_parts.append(hourly_stats.to_string(index=False))
    
    # === HOURLY REGION -> MODEL GROUPING ===
    result_parts.append("\n=== Hourly Region-wise -> Model-wise Analysis ===")
    
    # Group by datetime, region, model and calculate metrics
    hourly_region_model_stats = df.groupby(['datetime', 'region', 'modelId']).agg({
        'inputTokens': ['count', 'sum', 'mean', 'max', 'median'],
        'completionTokens': ['sum', 'mean', 'max', 'median'],
        'totalTokens': ['sum', 'mean', 'max', 'median']
    })
    
    # Flatten the column multi-index
    hourly_region_model_stats.columns = [f"{col[0]}_{col[1]}" for col in hourly_region_model_stats.columns]
    
    # Reset the index to get a flat dataframe
    hourly_region_model_stats = hourly_region_model_stats.reset_index()
    
    # Rename inputTokens_count to request_count
    hourly_region_model_stats = hourly_region_model_stats.rename(columns={'inputTokens_count': 'request_count'})
    
    # Format model IDs to be more readable
    hourly_region_model_stats['modelId'] = hourly_region_model_stats['modelId'].apply(
        lambda model: model.split('.')[-1] if '.' in model else model.split('/')[-1]
    )
    
    # Add the hourly region-model stats to result
    result_parts.append(hourly_region_model_stats.to_string(index=False))
    
    # Add summary statistics
    result_parts.append("\n=== Summary Statistics ===")
    
    # Total requests and tokens
    total_requests = hourly_stats['request_count'].sum()
    total_input_tokens = hourly_stats['inputTokens_sum'].sum()
    total_completion_tokens = hourly_stats['completionTokens_sum'].sum()
    total_tokens = hourly_stats['totalTokens_sum'].sum()
    
    result_parts.append(f"Total Requests: {total_requests:,}")
    result_parts.append(f"Total Input Tokens: {total_input_tokens:,}")
    result_parts.append(f"Total Completion Tokens: {total_completion_tokens:,}")
    result_parts.append(f"Total Tokens: {total_tokens:,}")
    
    # === REGION SUMMARY ===
    result_parts.append("\n=== Region Summary ===")
    region_summary = df.groupby('region').agg({
        'inputTokens': ['count', 'sum'],
        'completionTokens': ['sum'],
        'totalTokens': ['sum']
    })
    
    # Flatten region summary columns
    region_summary.columns = [f"{col[0]}_{col[1]}" for col in region_summary.columns]
    region_summary = region_summary.reset_index()
    region_summary = region_summary.rename(columns={'inputTokens_count': 'request_count'})
    
    result_parts.append(region_summary.to_string(index=False))
    
    # === MODEL SUMMARY ===
    result_parts.append("\n=== Model Summary ===")
    model_summary = df.groupby('modelId').agg({
        'inputTokens': ['count', 'sum'],
        'completionTokens': ['sum'],
        'totalTokens': ['sum']
    })
    
    # Flatten model summary columns
    model_summary.columns = [f"{col[0]}_{col[1]}" for col in model_summary.columns]
    model_summary = model_summary.reset_index()
    model_summary = model_summary.rename(columns={'inputTokens_count': 'request_count'})
    
    # Format model IDs to be more readable
    model_summary['modelId'] = model_summary['modelId'].apply(
        lambda model: model.split('.')[-1] if '.' in model else model.split('/')[-1]
    )
    
    result_parts.append(model_summary.to_string(index=False))
    
    # === USER SUMMARY ===
    if 'userId' in df.columns:
        result_parts.append("\n=== User Summary ===")
        user_summary = df.groupby('userId').agg({
            'inputTokens': ['count', 'sum'],
            'completionTokens': ['sum'],
            'totalTokens': ['sum']
        })
        
        # Flatten user summary columns
        user_summary.columns = [f"{col[0]}_{col[1]}" for col in user_summary.columns]
        user_summary = user_summary.reset_index()
        user_summary = user_summary.rename(columns={'inputTokens_count': 'request_count'})
        
        result_parts.append(user_summary.to_string(index=False))
        
    # === HOURLY REGION -> USER -> MODEL DETAILED SUMMARY ===
    if 'userId' in df.columns:
        result_parts.append("\n=== Hourly Region -> User -> Model Detailed Summary ===")
        hourly_region_user_model_summary = df.groupby(['datetime', 'region', 'userId', 'modelId']).agg({
            'inputTokens': ['count', 'sum', 'mean'],
            'completionTokens': ['sum', 'mean'],
            'totalTokens': ['sum', 'mean']
        })
        
        # Flatten columns
        hourly_region_user_model_summary.columns = [f"{col[0]}_{col[1]}" for col in hourly_region_user_model_summary.columns]
        hourly_region_user_model_summary = hourly_region_user_model_summary.reset_index()
        hourly_region_user_model_summary = hourly_region_user_model_summary.rename(columns={'inputTokens_count': 'request_count'})
        
        # Format model IDs to be more readable
        hourly_region_user_model_summary['modelId'] = hourly_region_user_model_summary['modelId'].apply(
            lambda model: model.split('.')[-1] if '.' in model else model.split('/')[-1]
        )
        
        result_parts.append(hourly_region_user_model_summary.to_string(index=False))
    
    # === HOURLY USAGE PATTERN ANALYSIS ===
    result_parts.append("\n=== Hourly Usage Pattern Analysis ===")
    
    # Group by hour of day (ignoring date) to see hourly patterns
    hour_pattern = df.groupby(df['timestamp'].dt.hour).agg({
        'inputTokens': ['count', 'sum'],
        'totalTokens': ['sum']
    })
    
    # Flatten hour pattern columns
    hour_pattern.columns = [f"{col[0]}_{col[1]}" for col in hour_pattern.columns]
    hour_pattern = hour_pattern.reset_index()
    hour_pattern = hour_pattern.rename(columns={
        'timestamp': 'hour_of_day',
        'inputTokens_count': 'request_count'
    })
    
    # Format the hour to be more readable
    hour_pattern['hour_of_day'] = hour_pattern['hour_of_day'].apply(
        lambda hour: f"{hour:02d}:00 - {hour:02d}:59"
    )
    
    result_parts.append(hour_pattern.to_string(index=False))
    
    # Combine all parts into a single string
    result = "\n".join(result_parts)
    
    return result

@mcp.tool()
async def get_ec2_spend_last_day(params: EC2Params) -> Dict[str, Any]:
    """
    Retrieve EC2 spend for the last day using standard AWS Cost Explorer API.

    By default, filters to RECORD_TYPE Usage so instance-type UnblendedCost aligns with
    Console-style on-demand usage (reservation recurring fees are excluded). Set
    ec2_record_type_filter to [] to include all record types.
    
    Returns:
        Dict[str, Any]: The raw response from the AWS Cost Explorer API, or None if an error occurs.
    """
    print(f"get_ec2_spend_last_day, params={params}")
    # Initialize the Cost Explorer client
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)

    
    # Calculate the time period - last day
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    ce_filter = _cost_explorer_dimension_filter(
        service_values=["Amazon Elastic Compute Cloud - Compute"],
        record_type_values=params.ec2_record_type_filter or None,
    )
    
    try:
        # Make the API call using get_cost_and_usage (standard API)
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Filter=ce_filter,
            Metrics=[
                'UnblendedCost',
                'UsageQuantity'
            ],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'INSTANCE_TYPE'
                }
            ]
        )
        
        # Process and print the results
        print(f"EC2 Spend from {start_date} to {end_date}:")
        print("-" * 50)
        
        total_cost = 0.0
        
        if 'ResultsByTime' in response and response['ResultsByTime']:
            time_period_data = response['ResultsByTime'][0]
            
            if 'Groups' in time_period_data:
                for group in time_period_data['Groups']:
                    instance_type = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    currency = group['Metrics']['UnblendedCost']['Unit']
                    usage = float(group['Metrics']['UsageQuantity']['Amount'])
                    
                    print(f"Instance Type: {instance_type}")
                    print(f"Cost: {cost:.4f} {currency}")
                    print(f"Usage: {usage:.2f}")
                    print("-" * 30)
                    
                    total_cost += cost
            
            # If no instance-level breakdown, show total
            if not time_period_data.get('Groups'):
                if 'Total' in time_period_data:
                    total = time_period_data['Total']
                    cost = float(total['UnblendedCost']['Amount'])
                    currency = total['UnblendedCost']['Unit']
                    print(f"Total EC2 Cost: {cost:.4f} {currency}")
                else:
                    print("No EC2 costs found for this period")
            else:
                print(f"Total EC2 Cost: {total_cost:.4f} {currency if 'currency' in locals() else 'USD'}")
                
            # Check if results are estimated
            if 'Estimated' in time_period_data:
                print(f"Note: These results are {'estimated' if time_period_data['Estimated'] else 'final'}")
        
        return response
        
    except Exception as e:
        print(f"Error retrieving EC2 cost data: {str(e)}")
        return None


@mcp.tool()
async def get_detailed_breakdown_by_day(params: EC2Params) -> str: #Dict[str, Any]:
    """
    Retrieve daily spend breakdown by region, service, and instance type.

    EC2 nested instance-type rows use ec2_record_type_filter (default RECORD_TYPE Usage)
    so costs are comparable to Console coverage / on-demand views; pass an empty list
    to include reservation fees and other record types in those rows.
    
    Args:
        params: Parameters specifying the number of days to look back
    
    Returns:
        Dict[str, Any]: A tuple containing:
            - A nested dictionary with cost data organized by date, region, and service
            - A string containing the formatted output report
        or (None, error_message) if an error occurs.
    """
    print(f"get_detailed_breakdown_by_day, params={params}")
    # Initialize the Cost Explorer client
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)
    
    # Get the days parameter
    days = params.days
    
    # Calculate the time period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Initialize output buffer
    output_buffer = []
    
    try:
        output_buffer.append(f"\nDetailed Cost Breakdown by Region, Service, and Instance Type ({days} days):")
        output_buffer.append("-" * 75)
        
        # First get the daily costs by region and service
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'REGION'
                },
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        # Create data structure to hold the results
        all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Process the results
        for time_data in response['ResultsByTime']:
            date = time_data['TimePeriod']['Start']
            
            output_buffer.append(f"\nDate: {date}")
            output_buffer.append("=" * 50)
            
            if 'Groups' in time_data and time_data['Groups']:
                # Create data structure for this date
                region_services = defaultdict(lambda: defaultdict(float))
                
                # Process groups
                for group in time_data['Groups']:
                    region, service = group['Keys']
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    currency = group['Metrics']['UnblendedCost']['Unit']
                    
                    region_services[region][service] = cost
                    all_data[date][region][service] = cost
                
                # Add the results for this date to the buffer
                for region in sorted(region_services.keys()):
                    output_buffer.append(f"\nRegion: {region}")
                    output_buffer.append("-" * 40)
                    
                    # Create a DataFrame for this region's services
                    services_df = pd.DataFrame({
                        'Service': list(region_services[region].keys()),
                        'Cost': list(region_services[region].values())
                    })
                    
                    # Sort by cost descending
                    services_df = services_df.sort_values('Cost', ascending=False)
                    
                    # Get top services by cost
                    top_services = services_df.head(5)
                    
                    # Add region's services table to buffer
                    output_buffer.append(tabulate(top_services.round(2), headers='keys', tablefmt='pretty', showindex=False))
                    
                    # If there are more services, indicate the total for other services
                    if len(services_df) > 5:
                        other_cost = services_df.iloc[5:]['Cost'].sum()
                        output_buffer.append(f"... and {len(services_df) - 5} more services totaling {other_cost:.2f} {currency}")
                    
                    # For EC2, get instance type breakdown
                    if any(s.startswith('Amazon Elastic Compute') for s in region_services[region].keys()):
                        try:
                            instance_response = get_instance_type_breakdown(
                                ce_client, 
                                date, 
                                region, 
                                'Amazon Elastic Compute Cloud - Compute', 
                                'INSTANCE_TYPE',
                                record_type_filter=params.ec2_record_type_filter or None,
                            )
                            
                            if instance_response:
                                output_buffer.append("\n  EC2 Instance Type Breakdown:")
                                output_buffer.append("  " + "-" * 38)
                                
                                # Get table with indentation
                                instance_table = tabulate(instance_response.round(2), headers='keys', tablefmt='pretty', showindex=False)
                                for line in instance_table.split('\n'):
                                    output_buffer.append(f"  {line}")
                        
                        except Exception as e:
                            output_buffer.append(f"  Note: Could not retrieve EC2 instance type breakdown: {str(e)}")
                    
                    # For SageMaker, get instance type breakdown
                    if any(s == 'Amazon SageMaker' for s in region_services[region].keys()):
                        try:
                            sagemaker_instance_response = get_instance_type_breakdown(
                                ce_client,
                                date,
                                region,
                                'Amazon SageMaker',
                                'INSTANCE_TYPE'
                            )
                            
                            if sagemaker_instance_response is not None and not sagemaker_instance_response.empty:
                                output_buffer.append("\n  SageMaker Instance Type Breakdown:")
                                output_buffer.append("  " + "-" * 38)
                                
                                # Get table with indentation
                                sagemaker_table = tabulate(sagemaker_instance_response.round(2), headers='keys', tablefmt='pretty', showindex=False)
                                for line in sagemaker_table.split('\n'):
                                    output_buffer.append(f"  {line}")
                            
                            # Also try to get usage type breakdown for SageMaker (notebooks, endpoints, etc.)
                            sagemaker_usage_response = get_instance_type_breakdown(
                                ce_client,
                                date,
                                region,
                                'Amazon SageMaker',
                                'USAGE_TYPE'
                            )
                            
                            if sagemaker_usage_response is not None and not sagemaker_usage_response.empty:
                                output_buffer.append("\n  SageMaker Usage Type Breakdown:")
                                output_buffer.append("  " + "-" * 38)
                                
                                # Get table with indentation
                                usage_table = tabulate(sagemaker_usage_response.round(2), headers='keys', tablefmt='pretty', showindex=False)
                                for line in usage_table.split('\n'):
                                    output_buffer.append(f"  {line}")
                        
                        except Exception as e:
                            output_buffer.append(f"  Note: Could not retrieve SageMaker breakdown: {str(e)}")
            else:
                output_buffer.append("No data found for this date")
            
            output_buffer.append("\n" + "-" * 75)
        
        # Join the buffer into a single string
        formatted_output = "\n".join(output_buffer)
        
        # Return both the raw data and the formatted output
        #return {"data": all_data, "formatted_output": formatted_output}
        return formatted_output
    
    except Exception as e:
        error_message = f"Error retrieving detailed breakdown: {str(e)}"
        #return {"data": None, "formatted_output": error_message}
        return error_message


@mcp.tool()
async def query_cost_explorer_cost_and_usage(params: CostExplorerQueryParams) -> Dict[str, Any]:
    """
    Generic AWS Cost Explorer query for cost and usage data.
    """
    print(f"query_cost_explorer_cost_and_usage, params={params}")
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=params.days)).strftime('%Y-%m-%d')

    request: Dict[str, Any] = {
        "TimePeriod": {
            "Start": start_date,
            "End": end_date
        },
        "Granularity": params.granularity,
        "Metrics": params.metrics
    }

    if params.group_by_dimension:
        request["GroupBy"] = [
            {
                "Type": "DIMENSION",
                "Key": params.group_by_dimension
            }
        ]

    ce_filter = _cost_explorer_dimension_filter(
        service_values=[params.service_filter] if params.service_filter else None,
        record_type_values=params.record_type_filter or None,
    )
    if ce_filter:
        request["Filter"] = ce_filter

    try:
        return ce_client.get_cost_and_usage(**request)
    except Exception as e:
        return {"error": f"Error querying Cost Explorer data: {str(e)}", "request": request}


def _get_reservation_utilization(params: ReservationParams, service_name: str) -> Dict[str, Any]:
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=params.days)).strftime('%Y-%m-%d')

    try:
        return ce_client.get_reservation_utilization(
            TimePeriod={
                "Start": start_date,
                "End": end_date
            },
            GroupBy=[
                {
                    "Type": "DIMENSION",
                    "Key": "SUBSCRIPTION_ID"
                }
            ],
            Granularity=params.granularity,
            Filter={
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": [service_name]
                }
            }
        )
    except Exception as e:
        return {"error": f"Error querying reservation utilization for {service_name}: {str(e)}"}


def _get_reservation_coverage(params: ReservationParams, service_name: str) -> Dict[str, Any]:
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=params.days)).strftime('%Y-%m-%d')

    try:
        return ce_client.get_reservation_coverage(
            TimePeriod={
                "Start": start_date,
                "End": end_date
            },
            GroupBy=[
                {
                    "Type": "DIMENSION",
                    "Key": "INSTANCE_TYPE"
                }
            ],
            Granularity=params.granularity,
            Filter={
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": [service_name]
                }
            }
        )
    except Exception as e:
        return {"error": f"Error querying reservation coverage for {service_name}: {str(e)}"}


@mcp.tool()
async def get_ec2_reservation_utilization(params: ReservationParams) -> Dict[str, Any]:
    """
    Retrieve EC2 reservation utilization from AWS Cost Explorer.
    """
    print(f"get_ec2_reservation_utilization, params={params}")
    return _get_reservation_utilization(params, "Amazon Elastic Compute Cloud - Compute")


@mcp.tool()
async def get_ec2_reservation_coverage(params: ReservationParams) -> Dict[str, Any]:
    """
    Retrieve EC2 reservation coverage from AWS Cost Explorer.
    """
    print(f"get_ec2_reservation_coverage, params={params}")
    return _get_reservation_coverage(params, "Amazon Elastic Compute Cloud - Compute")


@mcp.tool()
async def get_rds_reservation_utilization(params: ReservationParams) -> Dict[str, Any]:
    """
    Retrieve RDS reservation utilization from AWS Cost Explorer.
    """
    print(f"get_rds_reservation_utilization, params={params}")
    return _get_reservation_utilization(params, "Amazon Relational Database Service")


@mcp.tool()
async def get_rds_reservation_coverage(params: ReservationParams) -> Dict[str, Any]:
    """
    Retrieve RDS reservation coverage from AWS Cost Explorer.
    """
    print(f"get_rds_reservation_coverage, params={params}")
    return _get_reservation_coverage(params, "Amazon Relational Database Service")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _format_reservation_section(title: str, utilization: Dict[str, Any], coverage: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"\n=== {title} ===")

    if utilization.get("error"):
        lines.append(f"Utilization error: {utilization['error']}")
        return "\n".join(lines)
    if coverage.get("error"):
        lines.append(f"Coverage error: {coverage['error']}")
        return "\n".join(lines)

    util_total = utilization.get("Total", {})
    cov_total = coverage.get("Total", {})
    cov_hours = cov_total.get("CoverageHours", {})
    cov_cost = cov_total.get("CoverageCost", {})

    lines.append("Overall Reservation Snapshot:")
    lines.append(
        f"- Utilization: {_safe_float(util_total.get('UtilizationPercentage')):.2f}% | "
        f"Coverage: {_safe_float(cov_hours.get('CoverageHoursPercentage')):.2f}%"
    )
    lines.append(
        f"- Purchased Hours: {_safe_float(util_total.get('PurchasedHours')):.2f} | "
        f"Used Hours: {_safe_float(util_total.get('TotalActualHours')):.2f} | "
        f"Unused Hours: {_safe_float(util_total.get('UnusedHours')):.2f}"
    )
    lines.append(
        f"- Realized Savings: ${_safe_float(util_total.get('RealizedSavings')):.2f} | "
        f"Unrealized Savings: ${_safe_float(util_total.get('UnrealizedSavings')):.2f} | "
        f"On-Demand Cost (uncovered): ${_safe_float(cov_cost.get('OnDemandCost')):.2f}"
    )

    util_by_time = utilization.get("UtilizationsByTime", [])
    cov_by_time = coverage.get("CoveragesByTime", [])

    cov_by_start_date: Dict[str, Dict[str, Any]] = {}
    for cov_item in cov_by_time:
        period = cov_item.get("TimePeriod", {})
        start = period.get("Start")
        if start:
            cov_by_start_date[start] = cov_item.get("Total", {}).get("CoverageHours", {})

    rows: List[Dict[str, Any]] = []
    for util_item in util_by_time:
        period = util_item.get("TimePeriod", {})
        start = period.get("Start")
        if not start:
            continue
        util_total_for_day = util_item.get("Total", {})
        cov_hours_for_day = cov_by_start_date.get(start, {})
        rows.append(
            {
                "Date": start,
                "Utilization %": round(_safe_float(util_total_for_day.get("UtilizationPercentage")), 2),
                "Coverage %": round(_safe_float(cov_hours_for_day.get("CoverageHoursPercentage")), 2),
                "Used Hours": round(_safe_float(util_total_for_day.get("TotalActualHours")), 2),
                "Unused Hours": round(_safe_float(util_total_for_day.get("UnusedHours")), 2),
                "Realized Savings ($)": round(_safe_float(util_total_for_day.get("RealizedSavings")), 2),
            }
        )

    if rows:
        rows = sorted(rows, key=lambda item: item["Date"])
        trend_df = pd.DataFrame(rows)
        lines.append("\nDaily Trend:")
        lines.append(tabulate(trend_df, headers="keys", tablefmt="pretty", showindex=False))
    else:
        lines.append("No daily trend data returned for the requested period.")

    return "\n".join(lines)


@mcp.tool()
async def get_reservation_health_summary(params: ReservationSummaryParams) -> str:
    """
    Human-friendly reservation summary including utilization, coverage and daily trends.
    """
    print(f"get_reservation_health_summary, params={params}")
    services: List[tuple[str, str]] = []
    if params.service in ("EC2", "BOTH"):
        services.append(("EC2 Reservations", "Amazon Elastic Compute Cloud - Compute"))
    if params.service in ("RDS", "BOTH"):
        services.append(("RDS Reservations", "Amazon Relational Database Service"))

    report: List[str] = [
        f"Reservation Health Summary ({params.service})",
        f"Period: last {params.days} days | Granularity: {params.granularity}",
        "=" * 80,
    ]

    for section_title, aws_service_name in services:
        utilization = _get_reservation_utilization(params, aws_service_name)
        coverage = _get_reservation_coverage(params, aws_service_name)
        report.append(_format_reservation_section(section_title, utilization, coverage))

    return "\n".join(report)


# AWS normalization factors for regional Linux/UNIX instance size flexibility (units).
# Source: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ri-modifying.html
RI_FLEX_INSTANCE_SIZE_NORMALIZATION: Dict[str, float] = {
    "nano": 0.25,
    "micro": 0.5,
    "small": 1.0,
    "medium": 2.0,
    "large": 4.0,
    "xlarge": 8.0,
    "2xlarge": 16.0,
    "3xlarge": 24.0,
    "4xlarge": 32.0,
    "6xlarge": 48.0,
    "8xlarge": 64.0,
    "9xlarge": 72.0,
    "10xlarge": 80.0,
    "12xlarge": 96.0,
    "16xlarge": 128.0,
    "18xlarge": 144.0,
    "24xlarge": 192.0,
    "32xlarge": 256.0,
    "48xlarge": 384.0,
    "52xlarge": 416.0,
    "56xlarge": 448.0,
    "112xlarge": 896.0,
}

# Bare metal uses family-specific factors (same table as AWS docs).
RI_FLEX_METAL_NORMALIZATION: Dict[str, float] = {
    "a1.metal": 32.0,
    "c6g.metal": 128.0,
    "c6gd.metal": 128.0,
    "i3.metal": 128.0,
    "m6g.metal": 128.0,
    "m6gd.metal": 128.0,
    "r6g.metal": 128.0,
    "r6gd.metal": 128.0,
    "x2gd.metal": 128.0,
    "c5n.metal": 144.0,
    "c5.metal": 192.0,
    "c5d.metal": 192.0,
    "i3en.metal": 192.0,
    "m5.metal": 192.0,
    "m5d.metal": 192.0,
    "m5dn.metal": 192.0,
    "m5n.metal": 192.0,
    "r5.metal": 192.0,
    "r5b.metal": 192.0,
    "r5d.metal": 192.0,
    "r5dn.metal": 192.0,
    "r5n.metal": 192.0,
    "c6i.metal": 256.0,
    "c6id.metal": 256.0,
    "m6i.metal": 256.0,
    "m6id.metal": 256.0,
    "r6d.metal": 256.0,
    "r6id.metal": 256.0,
    "u-18tb1.metal": 448.0,
    "u-24tb1.metal": 448.0,
    "u-6tb1.metal": 896.0,
    "u-9tb1.metal": 896.0,
    "u-12tb1.metal": 896.0,
}


def _parse_instance_family_size(instance_type: str) -> tuple[str, str]:
    if "." not in instance_type:
        return instance_type, "unknown"
    family, size = instance_type.rsplit(".", 1)
    return family, size


def _ri_flexibility_normalization_units(instance_type: str) -> tuple[Optional[float], Optional[str]]:
    _family, size = _parse_instance_family_size(instance_type)
    if size == "unknown":
        return None, f"Unrecognized instance type format: {instance_type}"
    if size == "metal":
        factor = RI_FLEX_METAL_NORMALIZATION.get(instance_type)
        if factor is not None:
            return factor, None
        return None, f"Unknown bare metal type for normalization: {instance_type}"
    if size in RI_FLEX_INSTANCE_SIZE_NORMALIZATION:
        return RI_FLEX_INSTANCE_SIZE_NORMALIZATION[size], None
    m = re.match(r"^(\d+)xlarge$", size)
    if m:
        return float(int(m.group(1))) * 8.0, None
    return None, f"Unknown instance size token '{size}' on {instance_type}"


def _ri_product_platform_bucket(product_description: str) -> str:
    pd = product_description or ""
    if pd.startswith("Windows"):
        return "windows"
    if (
        pd.startswith("Linux")
        or pd.startswith("Red Hat")
        or pd.startswith("SUSE")
        or "Ubuntu" in pd
    ):
        return "linux"
    return "other"


def _instance_platform_bucket(platform_details: str) -> str:
    pd = platform_details or ""
    if pd.startswith("Windows"):
        return "windows"
    if (
        pd.startswith("Linux")
        or pd.startswith("Red Hat")
        or pd.startswith("SUSE")
        or pd.startswith("Ubuntu")
    ):
        return "linux"
    return "other"


def _platform_scope_matches(bucket: str, platform_scope: str) -> bool:
    if bucket == "other":
        return False
    if platform_scope == "all":
        return bucket in ("linux", "windows")
    if platform_scope == "linux_unix":
        return bucket == "linux"
    if platform_scope == "windows":
        return bucket == "windows"
    return False


@mcp.tool()
async def get_ec2_regional_ri_flexibility_report(params: EC2RIFlexibilityReportParams) -> str:
    """
    Compare active EC2 Reserved Instance capacity to running instances using AWS **normalization units**
    within each instance **family** (regional RIs with instance size flexibility).

    Simple per–instance-type counts (e.g. 7× c5.xlarge RIs vs 3× c5.xlarge running) ignore flexibility:
    a **c5.4xlarge** regional Linux RI can cover **2× c5.2xlarge** running in the same family. This report
    sums **purchased** and **running** footprints in the same unit space so those cases show as balanced.

    **Limitations**: Uses live `DescribeReservedInstances` / `DescribeInstances` (not Cost Explorer hour
    history). **Zonal** RIs do **not** get cross-size pooling here (see AZ section). Savings Plans,
    capacity blocks, and some marketplace OS RIs are not modeled. Windows flexibility rules differ;
    use `platform_scope` accordingly.
    """
    print(f"get_ec2_regional_ri_flexibility_report, params={params}")
    ec2 = get_aws_service_boto3_client("ec2", params.aws_account_id, params.region)

    warnings: List[str] = []

    # (platform_bucket, family, tenancy) -> normalization units
    regional_ri_units: Dict[tuple[str, str, str], float] = defaultdict(float)
    regional_run_units: Dict[tuple[str, str, str], float] = defaultdict(float)

    # Zonal: (az, instance_type, tenancy) -> reserved count vs running count (no cross-size flex)
    zonal_ri_count: Dict[tuple[str, str, str], int] = defaultdict(int)
    zonal_run_count: Dict[tuple[str, str, str], int] = defaultdict(int)

    try:
        ri_paginator = ec2.get_paginator("describe_reserved_instances")
        for page in ri_paginator.paginate(Filters=[{"Name": "state", "Values": ["active"]}]):
            for ri in page.get("ReservedInstances", []):
                product = ri.get("ProductDescription") or ""
                bucket = _ri_product_platform_bucket(product)
                if not _platform_scope_matches(bucket, params.platform_scope):
                    continue
                itype = ri.get("InstanceType") or ""
                tenancy = ri.get("InstanceTenancy") or "default"
                count = int(ri.get("InstanceCount") or 0)
                scope = ri.get("Scope") or ""
                units, warn = _ri_flexibility_normalization_units(itype)
                if warn:
                    warnings.append(warn)
                if units is None or count <= 0:
                    continue
                family, _size = _parse_instance_family_size(itype)
                if scope == "Region":
                    regional_ri_units[(bucket, family, tenancy)] += units * count
                elif scope == "Availability Zone":
                    az = ri.get("AvailabilityZone") or ""
                    zonal_ri_count[(az, itype, tenancy)] += count
    except Exception as e:
        return f"Error listing Reserved Instances: {e}"

    try:
        inst_paginator = ec2.get_paginator("describe_instances")
        for page in inst_paginator.paginate(
            Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
        ):
            for resv in page.get("Reservations", []):
                for inst in resv.get("Instances", []):
                    platform_details = inst.get("PlatformDetails") or ""
                    bucket = _instance_platform_bucket(platform_details)
                    if not _platform_scope_matches(bucket, params.platform_scope):
                        continue
                    itype = inst.get("InstanceType") or ""
                    if not itype:
                        continue
                    tenancy = (inst.get("Placement") or {}).get("Tenancy") or "default"
                    units, warn = _ri_flexibility_normalization_units(itype)
                    if warn:
                        warnings.append(warn)
                    if units is None:
                        continue
                    family, _size = _parse_instance_family_size(itype)
                    regional_run_units[(bucket, family, tenancy)] += units
                    az = (inst.get("Placement") or {}).get("AvailabilityZone") or ""
                    zonal_run_count[(az, itype, tenancy)] += 1
    except Exception as e:
        return f"Error listing running instances: {e}"

    lines: List[str] = [
        f"EC2 Regional RI flexibility vs running footprint — region {params.region}",
        f"Platform scope: {params.platform_scope}",
        "=" * 88,
        "",
        "Regional scope (instance size flexibility within family / tenancy / OS bucket):",
        "  Purchased RI units and Running units use the same AWS normalization scale (e.g. 1× 4xlarge = 2× 2xlarge).",
        "",
    ]

    keys = sorted(set(regional_ri_units.keys()) | set(regional_run_units.keys()))
    rows: List[Dict[str, Any]] = []
    for key in keys:
        bucket, family, tenancy = key
        ri_u = regional_ri_units.get(key, 0.0)
        run_u = regional_run_units.get(key, 0.0)
        if ri_u == 0 and run_u == 0:
            continue
        surplus = max(0.0, ri_u - run_u)
        gap = max(0.0, run_u - ri_u)
        if run_u > 0:
            pct_shape = round(100.0 * min(ri_u, run_u) / run_u, 1)
        else:
            pct_shape = None
        rows.append(
            {
                "OS": bucket,
                "Family": family,
                "Tenancy": tenancy,
                "RI units": round(ri_u, 2),
                "Running units": round(run_u, 2),
                "RI surplus": round(surplus, 2),
                "Run gap": round(gap, 2),
                "% shape covered": pct_shape if pct_shape is not None else "n/a",
            }
        )

    if rows:
        lines.append(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="pretty", showindex=False))
    else:
        lines.append("(No matching regional RI or running data for this scope.)")

    lines.extend(
        [
            "",
            "Zonal Reserved Instances (no cross-size pooling in this report):",
            "  Compares exact InstanceType in the RI's Availability Zone to running count in that AZ.",
            "",
        ]
    )

    z_keys = sorted(set(zonal_ri_count.keys()) | set(zonal_run_count.keys()))
    z_rows: List[Dict[str, Any]] = []
    for zk in z_keys:
        az, itype, tenancy = zk
        rc = zonal_ri_count.get(zk, 0)
        runc = zonal_run_count.get(zk, 0)
        if rc == 0 and runc == 0:
            continue
        z_rows.append(
            {
                "AZ": az,
                "InstanceType": itype,
                "Tenancy": tenancy,
                "Zonal RI qty": rc,
                "Running qty": runc,
                "Diff (run - RI)": runc - rc,
            }
        )

    if z_rows:
        lines.append(tabulate(pd.DataFrame(z_rows), headers="keys", tablefmt="pretty", showindex=False))
    else:
        lines.append("(No active zonal Reserved Instances in this account/region.)")

    lines.extend(
        [
            "",
            "Notes:",
            "- Regional Linux/UNIX Standard RIs typically share a normalization pool per instance family; this table reflects that.",
            "- For billing-hour accuracy (including credits and mixed purchases), prefer Cost Explorer reservation APIs.",
            "- Unknown instance sizes or bare metal types not in the mapping are skipped and listed below.",
        ]
    )

    if warnings:
        uniq = sorted(set(warnings))
        lines.append("")
        lines.append("Warnings / skipped items:")
        for w in uniq[:50]:
            lines.append(f"  - {w}")
        if len(uniq) > 50:
            lines.append(f"  ... and {len(uniq) - 50} more")

    return "\n".join(lines)


def get_instance_type_breakdown(
    ce_client,
    date: str,
    region: str,
    service: str,
    dimension_key: str,
    record_type_filter: Optional[List[str]] = None,
):
    """
    Helper function to get instance type or usage type breakdown for a specific service.
    
    Args:
        ce_client: The Cost Explorer client
        date: The date to query
        region: The AWS region
        service: The AWS service name
        dimension_key: The dimension to group by (e.g., 'INSTANCE_TYPE' or 'USAGE_TYPE')
        record_type_filter: Optional RECORD_TYPE dimension values (e.g. ['Usage'] for EC2
            on-demand-style costs). Omitted or empty: no RECORD_TYPE filter.
    
    Returns:
        DataFrame containing the breakdown or None if no data
    """
    tomorrow = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    and_clauses: List[Dict[str, Any]] = [
        {
            'Dimensions': {
                'Key': 'REGION',
                'Values': [region]
            }
        },
        {
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': [service]
            }
        },
    ]
    if record_type_filter:
        and_clauses.append(
            {
                'Dimensions': {
                    'Key': 'RECORD_TYPE',
                    'Values': record_type_filter,
                }
            }
        )

    instance_response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': date,
            'End': tomorrow
        },
        Granularity='DAILY',
        Filter={'And': and_clauses},
        Metrics=['UnblendedCost'],
        GroupBy=[
            {
                'Type': 'DIMENSION',
                'Key': dimension_key
            }
        ]
    )
    
    if ('ResultsByTime' in instance_response and 
        instance_response['ResultsByTime'] and 
        'Groups' in instance_response['ResultsByTime'][0] and 
        instance_response['ResultsByTime'][0]['Groups']):
        
        instance_data = instance_response['ResultsByTime'][0]
        instance_costs = []
        
        for instance_group in instance_data['Groups']:
            type_value = instance_group['Keys'][0]
            cost_value = float(instance_group['Metrics']['UnblendedCost']['Amount'])
            
            # Add a better label for the dimension used
            column_name = 'Instance Type' if dimension_key == 'INSTANCE_TYPE' else 'Usage Type'
            
            instance_costs.append({
                column_name: type_value,
                'Cost': cost_value
            })
        
        # Create DataFrame and sort by cost
        result_df = pd.DataFrame(instance_costs)
        if not result_df.empty:
            result_df = result_df.sort_values('Cost', ascending=False)
            return result_df
    
    return None


@mcp.tool()
async def get_compute_optimizer_ec2_recommendations(params: ComputeOptimizerEC2Params) -> Dict[str, Any]:
    """
    Retrieve AWS Compute Optimizer findings and recommendations for EC2 instances.
    """
    print(f"get_compute_optimizer_ec2_recommendations, params={params}")
    client = get_aws_service_boto3_client("compute-optimizer", params.aws_account_id, params.region)

    request: Dict[str, Any] = {
        "maxResults": params.max_results
    }
    if params.finding:
        request["filters"] = [
            {
                "name": "Finding",
                "values": [params.finding]
            }
        ]

    findings: List[Dict[str, Any]] = []
    next_token: Optional[str] = None

    try:
        while True:
            if next_token:
                request["nextToken"] = next_token
            response = client.get_ec2_instance_recommendations(**request)
            findings.extend(response.get("instanceRecommendations", []))
            next_token = response.get("nextToken")
            if not next_token:
                break

        return {
            "summary": {
                "finding_filter": params.finding,
                "total_recommendations": len(findings)
            },
            "recommendations": findings
        }
    except Exception as e:
        return {"error": f"Error querying Compute Optimizer EC2 recommendations: {str(e)}"}


@mcp.tool()
async def get_compute_optimizer_rds_recommendations(params: ComputeOptimizerRDSParams) -> Dict[str, Any]:
    """
    Retrieve AWS Compute Optimizer findings and recommendations for RDS DB instances.
    """
    print(f"get_compute_optimizer_rds_recommendations, params={params}")
    client = get_aws_service_boto3_client("compute-optimizer", params.aws_account_id, params.region)

    request: Dict[str, Any] = {
        "maxResults": params.max_results
    }
    if params.finding:
        request["filters"] = [
            {
                "name": "Finding",
                "values": [params.finding]
            }
        ]

    findings: List[Dict[str, Any]] = []
    next_token: Optional[str] = None

    try:
        while True:
            if next_token:
                request["nextToken"] = next_token
            response = client.get_rds_database_recommendations(**request)
            findings.extend(response.get("instanceRecommendations", []))
            next_token = response.get("nextToken")
            if not next_token:
                break

        return {
            "summary": {
                "finding_filter": params.finding,
                "total_recommendations": len(findings)
            },
            "recommendations": findings
        }
    except Exception as e:
        return {"error": f"Error querying Compute Optimizer RDS recommendations: {str(e)}"}


@mcp.tool()
async def get_cost_explorer_rightsizing_recommendations(params: RightsizingRecommendationParams) -> Dict[str, Any]:
    """
    Retrieve Cost Explorer rightsizing recommendations (EC2 only).
    """
    print(f"get_cost_explorer_rightsizing_recommendations, params={params}")
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)

    request: Dict[str, Any] = {
        "Service": params.service,
        "Configuration": {
            "RecommendationTarget": "SAME_INSTANCE_FAMILY",
            "BenefitsConsidered": True,
            "LookbackPeriodInDays": params.lookback_period
        },
        "PageSize": params.page_size
    }

    recommendations: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    try:
        while True:
            if next_page_token:
                request["NextPageToken"] = next_page_token
            response = ce_client.get_rightsizing_recommendation(**request)
            recommendations.extend(response.get("RightsizingRecommendations", []))
            next_page_token = response.get("NextPageToken")
            if not next_page_token:
                break

        return {
            "summary": {
                "service": params.service,
                "lookback_period": params.lookback_period,
                "total_recommendations": len(recommendations)
            },
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": f"Error querying Cost Explorer rightsizing recommendations: {str(e)}"}


@mcp.tool()
async def get_savings_plans_purchase_recommendations(params: SavingsPlansRecommendationParams) -> Dict[str, Any]:
    """
    Retrieve Cost Explorer Savings Plans purchase recommendations.
    """
    print(f"get_savings_plans_purchase_recommendations, params={params}")
    ce_client = get_aws_service_boto3_client("ce", params.aws_account_id, params.region)

    request: Dict[str, Any] = {
        "SavingsPlansType": "COMPUTE_SP",
        "TermInYears": params.term_in_years,
        "PaymentOption": params.payment_option,
        "LookbackPeriodInDays": params.lookback_period
    }

    try:
        response = ce_client.get_savings_plans_purchase_recommendation(**request)
        return {
            "summary": {
                "lookback_period": params.lookback_period,
                "payment_option": params.payment_option,
                "term_in_years": params.term_in_years
            },
            "recommendation_details": response.get("SavingsPlansPurchaseRecommendation", {})
        }
    except Exception as e:
        return {"error": f"Error querying Savings Plans purchase recommendations: {str(e)}"}


@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

def main():
    # Run the server with the specified transport
    mcp.run(transport=os.environ.get('MCP_TRANSPORT', 'stdio'))

if __name__ == "__main__":
    main()
