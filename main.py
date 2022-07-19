import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    download_output_name = "raw_data"
    download_output_fmt = 'parquet'
    download_artifact_name = f"{download_output_name}.{download_output_fmt}"
    if "download" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": download_artifact_name,
                "artifact_type": download_output_name,
                "artifact_description": "Data as downloaded"
            },
        )

    # preprocess
    preprocess_output_name = 'preprocessed_data'
    preprocess_output_fmt = 'csv'
    preprocess_artifact_name = f"{preprocess_output_name}.{preprocess_output_fmt}"
    if "preprocess" in steps_to_execute:
        preprocess_params = {"input_artifact": f"{download_artifact_name}:latest",
                             "artifact_name": preprocess_artifact_name,
                             "artifact_type": preprocess_output_name,
                             'artifact_description': 'Data with preprocessing applied'}

        _ = mlflow.run(os.path.join(root_path, 'preprocess'), "main", parameters=preprocess_params)

    if "check_data" in steps_to_execute:
        check_data_params = {'reference_artifact': config['data']['reference_dataset'],
                             'sample_artifact': f'{preprocess_artifact_name}:latest',
                             'ks_alpha': config['data']['ks_alpha']}
        _ = mlflow.run(os.path.join(root_path, 'check_data'), "main", parameters=check_data_params)

    # segregate
    segregate_output_name = 'data'  # this time is a prefix for train and test data
    segregate_output_type = 'segregated_data'
    if "segregate" in steps_to_execute:
        segregate_params = {'input_artifact': f"{preprocess_artifact_name}:latest",
                            'artifact_root': segregate_output_name,
                            'artifact_type': segregate_output_type,
                            'test_size': config['data']['test_size'],
                            'random_state': config['main']['random_seed'],
                            'stratify': config['data']['stratify']}

        _ = mlflow.run(os.path.join(root_path, 'segregate'), "main", parameters=segregate_params)

    # random forest
    train_data = f'{segregate_output_name}_train.csv:latest'
    rf_output_artifact = config['random_forest_pipeline']['export_artifact']
    if "random_forest" in steps_to_execute:
        # takes the arguments from the conf file - in the random forest section - to create a new one
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

            random_forest_params = {'train_data': train_data,
                                    'model_config': model_config,
                                    'export_artifact': rf_output_artifact,
                                    'random_seed': config['main']['random_seed'],
                                    'val_size': config['data']['val_size'],
                                    'stratify': config['data']['stratify']}

        _ = mlflow.run(os.path.join(root_path, 'random_forest'), 'main', parameters=random_forest_params)

    # evaluate
    test_data = f'{segregate_output_name}_test.csv:latest'
    model_input = f'{rf_output_artifact}:latest'
    if "evaluate" in steps_to_execute:
        evaluate_params = {'model_export': model_input,
                           'test_data': test_data}
        _ = mlflow.run(os.path.join(root_path, 'evaluate'), 'main', parameters=evaluate_params)


if __name__ == "__main__":
    go()
