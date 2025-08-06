def execute_pandas_expression(df, expression):
    """
    Safely evaluate a pandas expression like 'df["sleep"].mean()' and
    return a stringified result suitable for LLM-style answers.
    """
    import pandas as pd

    # Basic safety check
    if "__" in expression or ";" in expression or "import" in expression:
        return "Unsafe expression rejected."

    # Setup allowed environment
    local_vars = {"df": df, "pd": pd}
    try:
        expression = expression.strip("`").replace("python", "").strip()
        result = eval(expression, {"__builtins__": None}, local_vars)

        if isinstance(result, pd.Series):
            # Convert Series to "Column: Value" format
            # result = result.round(2).to_dict()
            if result.dtype.kind in 'fiu':  
                result = result.round(2)
            result = result.to_dict()

            return " ".join([f"{k}: [DP]{v}[/DP]" for k, v in result.items()])

        elif isinstance(result, pd.DataFrame):
            return "Result is a table (not supported for DP protection)."

        elif isinstance(result, (float, int)):
            return f"The result is [DP]{round(result, 2)}[/DP]."

        else:
            return str(result)

    except Exception as e:
        return f"Error executing query: {str(e)}"
