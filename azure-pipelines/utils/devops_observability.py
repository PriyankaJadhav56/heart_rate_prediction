import os
import sys
import time
import yaml
import requests
import json
from requests.structures import CaseInsensitiveDict

API_BASE_URL = sys.argv[1]
AZ_CLIENT_ID = sys.argv[2]
AZ_CLIENT_SECRET = sys.argv[3]
AZ_TENANT = sys.argv[4]
ENV = sys.argv[5]
GIT_REPO_URL = sys.argv[6]
COMMIT_ID = sys.argv[7]
TRIGGERED_BY = sys.argv[8]
JOB_TYPE = sys.argv[9]
PIPELINE_NAME = sys.argv[10]
PIPELINE_ID = sys.argv[11]
PIPELINE_RUN_ID = sys.argv[12]
TARGET_BRANCH = sys.argv[13]
DEVOPS_PROJECT_NAME = sys.argv[14]
DEVOPS_ORG_URL = sys.argv[15]
DATABRICKS_REPO_FOLDER_NAME = sys.argv[16]
RUN_ID = sys.argv[17]
JOB_ID = sys.argv[18]
DATABRICKS_HOST = sys.argv[19]
START_TIME = sys.argv[20]
DEVOPS_TOKEN = sys.argv[21]
SESSION_ID = sys.argv[22]

if ENV == "qa":
    DATABRICKS_REPO_FOLDER_NAME = f"{DATABRICKS_REPO_FOLDER_NAME}_QA"
    
print(f"API_BASE_URL : {API_BASE_URL}")
print(f"AZ_CLIENT_ID : {AZ_CLIENT_ID}")
print(f"DATABRICKS_REPO_FOLDER_NAME : {DATABRICKS_REPO_FOLDER_NAME}")
DEVOPS_ORG_NAME = DEVOPS_ORG_URL.split("/")[-2]
TARGET_BRANCH = TARGET_BRANCH.replace("refs/heads/", "")



def get_access_tokens(client_id, scope, client_secret, tenant_id):
    """
    Returns a bearer token
    """

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    data = {}
    data["client_id"] = client_id
    data["grant_type"] = "client_credentials"
    data["scope"] = scope
    data["client_secret"] = client_secret
    url = (
        "https://login.microsoftonline.com/" + tenant_id + "/oauth2/v2.0/token"
    )
    resp = requests.post(url, headers=headers, data=data).json()
    token = resp["access_token"]
    token_string = "Bearer" + " " + token
    return token_string

def get_headers(client_id, client_secret, tenant_id, target_Client_Id):
    h1 = CaseInsensitiveDict()
    scope = target_Client_Id + "/.default"
    h1["Authorization"] = get_access_tokens(
        client_id, scope, client_secret, tenant_id
    )
    h1["Content-Type"] = "application/json"
    return h1

def get_param(path, dict_object) :
    #general_configss/sdk_session_id/env
    path_tokens = path.split("/")
    lookup = dict_object
    for token in path_tokens :
        lookup = lookup.get(token, None)
        #print(token, lookup)
        if not lookup :
            return 
    return lookup


def get_project_details(API_BASE_URL,ENV,Header,SESSION_ID):
    yaml_file_path = "data_config/SolutionConfig.yaml"
    with open(yaml_file_path, 'r') as file:
        yaml_content = file.read()
    config = yaml.safe_load(yaml_content)
    if SESSION_ID == "None":
        session_id = get_param(f"general_configss/sdk_session_id/{ENV}")
    else:
        session_id = SESSION_ID
    
    url = f"https://{API_BASE_URL}/mlapi/get_transformation_session?dag_session_id={session_id}"
    try:
        print(f"url : {url}")
        response = requests.get(
            url,
            headers=Header,
        )
        return response.json()
    except Exception as e:
        print(e)
        return None
    
def get_devops_folder_name(DEVOPS_TOKEN):
    url = f"https://dev.azure.com/{DEVOPS_ORG_NAME}/{DEVOPS_PROJECT_NAME}/_apis/pipelines/{PIPELINE_ID}?api-version=7.1-preview.1"
    try:
        response = requests.get(url, auth=("", DEVOPS_TOKEN))
        result = response.json()["folder"][1:]
        return result
    except Exception as e:
        print(e)
    


def devops_data_add(API_BASE_URL,PAYLOAD,Header):
    url = f"https://{API_BASE_URL}/mlapi/devops/data/add"
    try:
        print(f"url : {url}")
        print(f"payload : {PAYLOAD}")
        print(type(PAYLOAD))
        status = requests.post(
            url,
            json=PAYLOAD,
            headers=Header,
        )
        print(f"status code {status}")
        print(status.json())
    except Exception as e:
        print(e)


# def get_target_env_details():
def get_target_params():
    yaml_file_path = "data_config/SolutionConfig.yaml"
    with open(yaml_file_path, 'r') as file:
        yaml_content = file.read()
    config = yaml.safe_load(yaml_content)
    tracking_env =  config.get("general_configs").get("tracking_env")
    ENV = tracking_env
    tracking_url = config.get("general_configs").get("tracking_url")
    if tracking_url:
        API_BASE_URL = tracking_url
    else:
        API_BASE_URL = os.environ.get(f"API_BASE_URL_{ENV.upper()}")
    Target_CLIENT_ID =  os.environ.get(f"AZ_CLIENT_ID_{ENV.upper()}")
    SESSION_ID = config.get("general_configs").get("sdk_session_id").get(f"{ENV}")
    print(ENV, API_BASE_URL, Target_CLIENT_ID, SESSION_ID)
    return ENV, API_BASE_URL, Target_CLIENT_ID, SESSION_ID


ENV, API_BASE_URL, Target_CLIENT_ID , SESSION_ID = get_target_params()

Header = get_headers(AZ_CLIENT_ID, AZ_CLIENT_SECRET, AZ_TENANT, Target_CLIENT_ID)

session_info = get_project_details(API_BASE_URL,ENV,Header,SESSION_ID)
session_data = session_info.get('data', {})
state_dict = session_data.get('state_dict', {})

DEVOPS_FOLDER= get_devops_folder_name(DEVOPS_TOKEN)

PAYLOAD = {
            "project_id": state_dict.get('project_id'),
            "project_name": state_dict.get('project_name'),
            "version": state_dict.get('version'),
            "job_id": JOB_ID,
            "run_id": RUN_ID,
            "job_type": JOB_TYPE,
            "git_hub_repo_url": GIT_REPO_URL,
            "databricks_repo_folder_name": DATABRICKS_REPO_FOLDER_NAME,
            "devops_organization_name": DEVOPS_ORG_NAME,
            "devops_project_name": DEVOPS_PROJECT_NAME,
            "pipeline_name": PIPELINE_NAME,
            "pipeline_definition_id": PIPELINE_ID,
            "build_id": PIPELINE_RUN_ID,
            "commit_id": COMMIT_ID,
            "commit_url": f"{GIT_REPO_URL}/commit/{COMMIT_ID}",
            "target_branch_name": TARGET_BRANCH,
            "build_url": f"{DEVOPS_ORG_URL}{DEVOPS_PROJECT_NAME}/_build/results?buildId={PIPELINE_RUN_ID}",
            "triggered_by": TRIGGERED_BY,
            "start_time": START_TIME,
            "run_notebook_url":f"{DATABRICKS_HOST}/jobs/{JOB_ID}/runs/{RUN_ID}",
            "devops_orchestrator": "Azure Devops",
            "devops_badge" : f"https://dev.azure.com/mlops-tiger/mlcore/_apis/build/status/{DEVOPS_FOLDER}/{PIPELINE_NAME}?branchName={TARGET_BRANCH}",
            "devops_version" : "v1"
    }

print(PAYLOAD)
devops_data_add(API_BASE_URL,PAYLOAD, Header)