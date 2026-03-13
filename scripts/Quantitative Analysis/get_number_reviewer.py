import re
import math
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
EXCEL_FILE = "Pure human.xlsx"
OUTPUT_FILE = "pure_human_with_reviewers.xlsx"
SUMMARY_FILE = "reviewer_stats_summary.txt"

# Put your GitHub personal access token here
GITHUB_TOKEN = ""

# If True, bot accounts like xxx[bot] will be excluded
EXCLUDE_BOTS = False

# Request timeout
REQUEST_TIMEOUT = 30


# =========================
# HELPERS
# =========================
def parse_pr_url(url: str):
    """
    Parse GitHub PR URL like:
    https://github.com/owner/repo/pull/123
    """
    if pd.isna(url):
        return None

    url = str(url).strip()
    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not m:
        return None

    owner, repo, pr_number = m.group(1), m.group(2), m.group(3)
    return owner, repo, pr_number


def github_get_all_pages(url: str, headers: dict):
    """
    Fetch all pages from a GitHub paginated endpoint.
    """
    results = []
    while url:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            results.extend(data)
        else:
            # for safety, though /reviews should return a list
            results.append(data)

        # Parse Link header for next page
        next_url = None
        link = resp.headers.get("Link", "")
        if link:
            parts = link.split(",")
            for part in parts:
                section = part.strip()
                if 'rel="next"' in section:
                    match = re.search(r'<([^>]+)>', section)
                    if match:
                        next_url = match.group(1)
                        break
        url = next_url

    return results


def get_unique_reviewer_count(pr_url: str, headers: dict, exclude_bots: bool = False):
    """
    Count unique reviewers from:
    GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews

    Reviewer is defined as a user who submitted at least one formal review.
    """
    parsed = parse_pr_url(pr_url)
    if not parsed:
        return None, "Invalid PR url"

    owner, repo, pr_number = parsed
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews?per_page=100"

    try:
        reviews = github_get_all_pages(api_url, headers=headers)

        reviewers = set()
        for review in reviews:
            user = review.get("user")
            if not user:
                continue

            login = user.get("login")
            if not login:
                continue

            if exclude_bots and login.endswith("[bot]"):
                continue

            reviewers.add(login)

        return len(reviewers), None

    except requests.HTTPError as e:
        return None, f"HTTPError: {e}"
    except requests.RequestException as e:
        return None, f"RequestException: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def compute_iqr_bounds(series: pd.Series):
    """
    Compute Q1, Q3, IQR, lower bound, upper bound
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def descriptive_stats(series: pd.Series):
    """
    Return Min, Q1, Median, Mean, Q3, Max
    """
    return {
        "Min": float(series.min()),
        "Q1": float(series.quantile(0.25)),
        "Median": float(series.median()),
        "Mean": float(series.mean()),
        "Q3": float(series.quantile(0.75)),
        "Max": float(series.max()),
    }


def format_stats(stats: dict, digits: int = 2):
    formatted = {}
    for k, v in stats.items():
        if pd.isna(v):
            formatted[k] = "NA"
        else:
            if float(v).is_integer():
                formatted[k] = str(int(v))
            else:
                formatted[k] = f"{v:.{digits}f}"
    return formatted


# =========================
# MAIN
# =========================
def main():
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # Read Excel
    df = pd.read_excel(EXCEL_FILE)
    if "PR url" not in df.columns:
        raise ValueError("Column 'PR url' not found in the Excel file.")

    reviewer_counts = []
    error_messages = []

    print(f"Processing {len(df)} rows...")

    for idx, pr_url in enumerate(df["PR url"], start=1):
        print(f"[{idx}/{len(df)}] {pr_url}")

        if pd.isna(pr_url) or str(pr_url).strip() == "":
            reviewer_counts.append(None)
            error_messages.append("Empty PR url")
            continue

        count, err = get_unique_reviewer_count(
            pr_url=str(pr_url),
            headers=headers,
            exclude_bots=EXCLUDE_BOTS
        )
        reviewer_counts.append(count)
        error_messages.append(err)

    # Save reviewer counts back to Excel
    df["Num reviewers"] = reviewer_counts
    df["Reviewer extraction error"] = error_messages
    df.to_excel(OUTPUT_FILE, index=False)

    # Keep only valid numeric reviewer counts
    valid_series = pd.to_numeric(df["Num reviewers"], errors="coerce").dropna()

    if len(valid_series) == 0:
        raise ValueError("No valid reviewer counts were extracted.")

    # Original stats
    original_stats = descriptive_stats(valid_series)
    q1, q3, iqr, lower, upper = compute_iqr_bounds(valid_series)

    # Remove outliers using IQR rule
    filtered_series = valid_series[(valid_series >= lower) & (valid_series <= upper)]

    if len(filtered_series) == 0:
        raise ValueError("All values were removed as outliers. Please inspect the data.")

    filtered_stats = descriptive_stats(filtered_series)

    original_fmt = format_stats(original_stats)
    filtered_fmt = format_stats(filtered_stats)

    outlier_values = valid_series[(valid_series < lower) | (valid_series > upper)]
    num_outliers = len(outlier_values)

    summary_lines = [
        "Reviewer Count Statistics",
        "=========================",
        f"Input file: {EXCEL_FILE}",
        f"Output file with reviewer counts: {OUTPUT_FILE}",
        f"Exclude bot reviewers: {EXCLUDE_BOTS}",
        "",
        "Original data",
        "-------------",
        f"N = {len(valid_series)}",
        f"Min = {original_fmt['Min']}",
        f"Q1 = {original_fmt['Q1']}",
        f"Median = {original_fmt['Median']}",
        f"Mean = {original_fmt['Mean']}",
        f"Q3 = {original_fmt['Q3']}",
        f"Max = {original_fmt['Max']}",
        "",
        "IQR outlier rule",
        "----------------",
        f"IQR = Q3 - Q1 = {q3:.4f} - {q1:.4f} = {iqr:.4f}",
        f"Lower bound = Q1 - 1.5 * IQR = {lower:.4f}",
        f"Upper bound = Q3 + 1.5 * IQR = {upper:.4f}",
        f"Number of outliers removed = {num_outliers}",
        f"Outlier values = {list(outlier_values.values)}",
        "",
        "After outlier removal",
        "---------------------",
        f"N = {len(filtered_series)}",
        f"Min = {filtered_fmt['Min']}",
        f"Q1 = {filtered_fmt['Q1']}",
        f"Median = {filtered_fmt['Median']}",
        f"Mean = {filtered_fmt['Mean']}",
        f"Q3 = {filtered_fmt['Q3']}",
        f"Max = {filtered_fmt['Max']}",
        "",
        "Paper-ready row",
        "---------------",
        f"Min & Q1 & Median & Mean & Q3 & Max = "
        f"{filtered_fmt['Min']} & {filtered_fmt['Q1']} & {filtered_fmt['Median']} & "
        f"{filtered_fmt['Mean']} & {filtered_fmt['Q3']} & {filtered_fmt['Max']}",
    ]

    summary_text = "\n".join(summary_lines)

    # with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    #     f.write(summary_text)

    print("\n" + summary_text)


if __name__ == "__main__":
    main()