from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[3]
OUTPUT_ROOT=REPO_ROOT/"outputs"


def model_output_dir(model):
    return OUTPUT_ROOT/model.group/model.model_id


def stage_dir(model,stage:str):
    p=model_output_dir(model)/stage
    p.mkdir(parents=True,exist_ok=True)
    return p


def stage_file(model,stage:str,content:str,extension:str):
    ext=extension if extension.startswith(".") else "."+extension
    return stage_dir(model,stage)/f"{stage}__{content}{ext}"


def training_run_dir(model,condition:str,repeat:int):
    p=stage_dir(model,"s3_train")/condition/f"repeat{int(repeat):02d}"
    p.mkdir(parents=True,exist_ok=True)
    return p


def training_run_file(model,condition:str,repeat:int,content:str,extension:str):
    ext=extension if extension.startswith(".") else "."+extension
    return training_run_dir(model,condition,repeat)/f"s3_train__{content}{ext}"


def rotation_root_dir(model):
    p=model_output_dir(model)/"rotation"
    p.mkdir(parents=True,exist_ok=True)
    return p


def rotation_shared_run_dir(model,stage:str,condition:str,repeat:int):
    p=rotation_root_dir(model)/stage/condition/f"repeat{int(repeat):02d}"
    p.mkdir(parents=True,exist_ok=True)
    return p


def rotation_shared_run_file(model,stage:str,condition:str,repeat:int,content:str,extension:str):
    ext=extension if extension.startswith(".") else "."+extension
    return rotation_shared_run_dir(model,stage,condition,repeat)/f"{stage}__{content}{ext}"


def rotation_attempt_run_dir(model,scheme:str,version:str,stage:str,condition:str,repeat:int):
    p=(
        rotation_root_dir(model)
        /model.group
        /scheme
        /version
        /stage
        /condition
        /f"repeat{int(repeat):02d}"
    )
    p.mkdir(parents=True,exist_ok=True)
    return p


def rotation_attempt_run_file(model,scheme:str,version:str,stage:str,condition:str,repeat:int,content:str,extension:str):
    ext=extension if extension.startswith(".") else "."+extension
    return rotation_attempt_run_dir(
        model,scheme,version,stage,condition,repeat
    )/f"{stage}__{content}{ext}"
