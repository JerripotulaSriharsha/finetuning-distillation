from kfp.components import load_component_from_file
import kfp.dsl as dsl
from kfp import compiler
import argparse
from kfp import Client
from typing import Optional
# Import the kubernetes extension
from kfp import kubernetes


sft_model_train_op = load_component_from_file(file_path="ops/train/SFT/model_train.yaml")
kd_model_train_op = load_component_from_file(file_path="ops/train/KD/model_train.yaml")
dpo_model_train_op = load_component_from_file(file_path="ops/train/DPO/model_train.yaml")
# model_infer_codegen_op = load_component_from_file(file_path="ops/infer/codegen/model_infer.yaml")
model_infer_idp_op = load_component_from_file(file_path="ops/infer/idp/model_infer.yaml")
eval_model_op = load_component_from_file("ops/eval/model_eval.yaml")
model_infer_codegen_op = load_component_from_file(file_path="ops/infer/codegen/model_infer.yaml")

def create_pipeline(student : str, data:str, save_dir:str, epochs:int, prompt:Optional[str], 
                   flow_type:str, train_type:str, schema:Optional[str], document:Optional[str],eval_data:Optional[str],val_per_class:Optional[int],offload_folder:Optional[str]):
    
    # Pre-process optional parameters to provide default empty strings.
    default_schema = schema if schema is not None else ""
    default_document = document if document is not None else ""
    default_prompt = prompt if prompt is not None else ""
    default_offload_folder = offload_folder if offload_folder is not None else ""
    print("Hey ")

    @dsl.pipeline(name="Finetune Pipeline", description="Finetuning pipeline")
    def pipeline(student:str=student, data:str=data, save_dir:str=save_dir, epochs:int=epochs, 
                flow_type:str=flow_type, train_type:str=train_type, schema:str=default_schema, 
                document:str=default_document, prompt:str=default_prompt,eval_data:str=eval_data,val_per_class:int=val_per_class,offload_folder:str=default_offload_folder):
        
        # --- STEP 1: CREATE THE PVC ONCE FOR THE WHOLE PIPELINE ---
        # This is more efficient as we only need one shared storage space.
        pvc_name = 'model-storage'
        pvc_task = kubernetes.CreatePVC(
            pvc_name_suffix=pvc_name,
            access_modes=['ReadWriteMany'],
            size='10Gi',
            storage_class_name='standard',  # IMPORTANT: Change to your cluster's storage class
        )
        
        # --- STEP 2: DEFINE TASKS THAT WILL USE THE PVC ---
        
        # We will collect the final tasks to schedule cleanup after them
        final_tasks = []

        # First level: Flow type selection
        with dsl.If(flow_type == "IDP", name="IDP Flow"):
            student_model = "Qwen/Qwen2.5-3B-Instruct"
            
            # Second level: Training type selection
            with dsl.If(train_type == "SFT", name="SFT Training"):
                model_trainer = sft_model_train_op(
                    student=student_model, 
                    data=data, 
                    save_dir="/mnt/model", 
                    epochs=epochs
                ).set_display_name("SFT Model Training").set_caching_options(enable_caching=False).set_env_variable( name="MLFLOW_TRACKING_URI",value="http://mlflow-service.mlflow.svc.cluster.local:5000")  # Disable caching
                
                model_trainer.set_memory_request("2Gi")
                model_trainer.set_memory_limit("6Gi")
                model_trainer.set_cpu_request("1")
                model_trainer.set_cpu_limit("2")

                kubernetes.mount_pvc(
                    model_trainer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model",
                )

                model_eval = eval_model_op(
                    model_dir="/mnt/model",
                    data=eval_data,
                    training_type="sft",
                    base_model=student_model,
                    max_len=2048,
                    max_new_tokens=512,
                    val_per_class=val_per_class,
                    offload_folder=offload_folder,
                    idp_mode=True,
                    mlflow_experiment_name="sft_evaluation"
                ).set_display_name("Model Eval").set_caching_options(enable_caching=False)

                model_eval.after(model_trainer)
                kubernetes.mount_pvc(model_eval, pvc_name=pvc_name, mount_path="/mnt/model")
                
                model_infer = model_infer_idp_op(
                    model_dir="/mnt/model", 
                    schema=schema, 
                    document=document
                ).set_display_name("IDP Model Inference").set_caching_options(enable_caching=False)  # Disable caching
                
                kubernetes.mount_pvc(
                    model_infer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )
                model_infer.after(model_eval)
                final_tasks.append(model_infer)
                
            with dsl.If(train_type == "KD", name="KD Training"):
                model_trainer = kd_model_train_op(
                    student=student_model, 
                    data=data, 
                    save_dir="/mnt/model", 
                    epochs=epochs
                ).set_display_name("KD Model Training").set_caching_options(enable_caching=False).set_env_variable( name="MLFLOW_TRACKING_URI",value="http://mlflow-service.mlflow.svc.cluster.local:5000")   # Disable caching

                
                kubernetes.mount_pvc(
                    model_trainer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )

                model_eval = eval_model_op(
                    model_dir="/mnt/model",
                    data=eval_data,
                    training_type="kd",
                    base_model=student_model,
                    max_len=2048,
                    max_new_tokens=512,
                    val_per_class=val_per_class,
                    offload_folder=offload_folder,
                    idp_mode=True,
                    mlflow_experiment_name="kd_evaluation"
                ).set_display_name("Model Eval").set_caching_options(enable_caching=False)

                model_eval.after(model_trainer)
                kubernetes.mount_pvc(model_eval, pvc_name=pvc_name, mount_path="/mnt/model")
                
                model_infer = model_infer_idp_op(
                    model_dir="/mnt/model", 
                    schema=schema, 
                    document=document
                ).set_display_name("IDP Model Inference").set_caching_options(enable_caching=False)  # Disable caching
                
                kubernetes.mount_pvc(
                    model_infer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )
                model_infer.after(model_eval)
                final_tasks.append(model_infer)
                
            with dsl.If(train_type == "DPO", name="DPO Training"):
                model_trainer = dpo_model_train_op(
                    student=student_model, 
                    data=data, 
                    save_dir="/mnt/model", 
                    epochs=epochs,
                ).set_display_name("DPO Model Training").set_caching_options(enable_caching=False).set_env_variable( name="MLFLOW_TRACKING_URI",value="http://mlflow-service.mlflow.svc.cluster.local:5000")   # Disable caching
                
                kubernetes.mount_pvc(
                    model_trainer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )

                model_eval = eval_model_op(
                    model_dir="/mnt/model",
                    data=eval_data,
                    training_type="dpo",
                    base_model=student_model,
                    max_len=2048,
                    max_new_tokens=512,
                    val_per_class=val_per_class,
                    offload_folder=offload_folder,
                    idp_mode=True,
                    mlflow_experiment_name="dpo_evaluation"
                ).set_display_name("Model Eval").set_caching_options(enable_caching=False)

                model_eval.after(model_trainer)
                kubernetes.mount_pvc(model_eval, pvc_name=pvc_name, mount_path="/mnt/model")
                
                model_infer = model_infer_idp_op(
                    model_dir="/mnt/model", 
                    schema=schema, 
                    document=document
                ).set_display_name("IDP Model Inference").set_caching_options(enable_caching=False)  # Disable caching
                
                kubernetes.mount_pvc(
                    model_infer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )
                model_infer.after(model_eval)
                final_tasks.append(model_infer)

        with dsl.If(flow_type == "CodeGen", name="CodeGen Flow"):
            student_model = "Qwen/Qwen2.5-Coder-3B"
            
            # Second level: Training type selection
            with dsl.If(train_type == "SFT", name="SFT Training"):
                model_trainer = sft_model_train_op(
                    student=student_model, 
                    data=data, 
                    save_dir="/mnt/model", 
                    epochs=epochs
                ).set_display_name("SFT Model Training").set_caching_options(enable_caching=False).set_env_variable( name="MLFLOW_TRACKING_URI",value="http://mlflow-service.mlflow.svc.cluster.local:5000")   # Disable caching
                
                kubernetes.mount_pvc(
                    model_trainer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )

                model_eval = eval_model_op(
                    model_dir="/mnt/model",
                    data=eval_data,
                    training_type="sft",
                    base_model=student_model,
                    max_len=2048,
                    max_new_tokens=512,
                    val_per_class=val_per_class,
                    offload_folder=offload_folder,
                    idp_mode=False,
                    mlflow_experiment_name="sft_evaluation"
                ).set_display_name("Model Eval").set_caching_options(enable_caching=False)

                model_eval.after(model_trainer)
                kubernetes.mount_pvc(model_eval, pvc_name=pvc_name, mount_path="/mnt/model")

                                
                model_infer = model_infer_codegen_op(
                    model_dir="/mnt/model", 
                    prompt=prompt
                ).set_display_name("CodeGen Model Inference").set_caching_options(enable_caching=False)  # Disable caching
                
                kubernetes.mount_pvc(
                    model_infer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )
                model_infer.after(model_eval)
                final_tasks.append(model_infer)
                
            with dsl.If(train_type == "KD", name="KD Training"):
                model_trainer = kd_model_train_op(
                    student=student_model, 
                    data=data, 
                    save_dir="/mnt/model", 
                    epochs=epochs
                ).set_display_name("KD Model Training").set_caching_options(enable_caching=False).set_env_variable( name="MLFLOW_TRACKING_URI",value="http://mlflow-service.mlflow.svc.cluster.local:5000")   # Disable caching
                
                kubernetes.mount_pvc(
                    model_trainer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )

                model_eval = eval_model_op(
                    model_dir="/mnt/model",
                    data=eval_data,
                    training_type="kd",
                    base_model=student_model,
                    max_len=2048,
                    max_new_tokens=512,
                    val_per_class=val_per_class,
                    offload_folder=offload_folder,
                    idp_mode=False,
                    mlflow_experiment_name="kd_evaluation"
                ).set_display_name("Model Eval").set_caching_options(enable_caching=False)

                model_eval.after(model_trainer)
                kubernetes.mount_pvc(model_eval, pvc_name=pvc_name, mount_path="/mnt/model")
                
                model_infer = model_infer_codegen_op(
                    model_dir="/mnt/model", 
                    prompt=prompt
                ).set_display_name("CodeGen Model Inference").set_caching_options(enable_caching=False)  # Disable caching
                
                kubernetes.mount_pvc(
                    model_infer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )
                model_infer.after(model_eval)
                final_tasks.append(model_infer)
                
            with dsl.If(train_type == "DPO", name="DPO Training"):
                model_trainer = dpo_model_train_op(
                    student=student_model, 
                    data=data, 
                    save_dir="/mnt/model", 
                    epochs=epochs
                ).set_display_name("DPO Model Training").set_caching_options(enable_caching=False).set_env_variable( name="MLFLOW_TRACKING_URI",value="http://mlflow-service.mlflow.svc.cluster.local:5000")   # Disable caching
                
                kubernetes.mount_pvc(
                    model_trainer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )

                model_eval = eval_model_op(
                    model_dir="/mnt/model",
                    data=eval_data,
                    training_type="dpo",
                    base_model=student_model,
                    max_len=2048,
                    max_new_tokens=512,
                    val_per_class=val_per_class,
                    offload_folder=offload_folder,
                    idp_mode=False,
                    mlflow_experiment_name="dpo_evaluation"
                ).set_display_name("Model Eval").set_caching_options(enable_caching=False)

                model_eval.after(model_trainer)
                kubernetes.mount_pvc(model_eval, pvc_name=pvc_name, mount_path="/mnt/model")
                
                model_infer = model_infer_codegen_op(
                    model_dir="/mnt/model", 
                    prompt=prompt
                ).set_display_name("CodeGen Model Inference").set_caching_options(enable_caching=False)  # Disable caching
                
                kubernetes.mount_pvc(
                    model_infer,
                    pvc_name=pvc_name,
                    mount_path="/mnt/model"
                )
                model_infer.after(model_eval)
                final_tasks.append(model_infer)
        
        # --- STEP 3: CLEAN UP THE PVC AFTER ALL WORK IS DONE ---
        # This ensures the PVC is deleted no matter which branch was taken.
        # The .after() call with a list makes the task wait for all tasks in the list to complete.
        # kubernetes.DeletePVC(pvc_name=pvc_task.outputs['name']).after(*final_tasks)
    
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--student",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Data_path"
    )

    parser.add_argument(
        "--host", 
        type=str, 
        default="http://127.0.0.1:8080/",
        help="Kubeflow Pipelines API endpoint", 
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/model",
        help="Save dir path (will be mounted as a persistent volume)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="No of epochs"
    )

    parser.add_argument(
        "--prompt",
        type= str,
        default="",
        help = "Prompt"
    )

    parser.add_argument(
        "--flow_type",
        type= str,
        required=True,
        choices=["IDP", "CodeGen"],
        help = "IDP or CodeGen"
    )

    parser.add_argument(
        "--train_type",
        type= str,
        required=True,
        choices=["SFT", "KD", "DPO"],
        help = "SFT, KD or DPO"
    )

    parser.add_argument(
        "--schema",
        type= str,
        default="",
        help = "schema"
    )

    parser.add_argument(
        "--document",
        type= str,
        default="",
        help = "document"
    )

    parser.add_argument(
        "--eval_data",
        type= str,
        default="",
        help = "Eval data"
    )

    parser.add_argument(
        "--val_per_class",
        type= int,
        default=0,
        help = "val_per_class"
    )
    
    parser.add_argument(
        "--offload_folder",
        type= str,
        default="",
        help = "offload_folder"
    )

    input_args = parser.parse_args()

    pipeline_func = create_pipeline(
        input_args.student,
        input_args.data,
        input_args.save_dir,
        input_args.epochs,
        input_args.prompt,
        input_args.flow_type,
        input_args.train_type,
        input_args.schema,
        input_args.document,
        input_args.eval_data,
        input_args.val_per_class,
        input_args.offload_folder,
    )

    pipeline_filename = "sample_pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=pipeline_filename,
    )

    # Connect to Kubeflow Pipelines
    client = Client(host=input_args.host)

    # Upload pipeline
    client.upload_pipeline(pipeline_filename, pipeline_name="Training Test")

    # Optionally run immediately
    client.create_run_from_pipeline_func(
        pipeline_func=pipeline_func,
        arguments={
            "student": input_args.student,
            "data": input_args.data,
            "save_dir": input_args.save_dir,
            "epochs": input_args.epochs,
            "prompt": input_args.prompt,
            "flow_type": input_args.flow_type,
            "train_type": input_args.train_type,
            "schema": input_args.schema,
            "document": input_args.document,
            "eval_data": input_args.eval_data,
            "val_per_class":input_args.val_per_class,
            "offload_folder":input_args.offload_folder
        },
        enable_caching=False  # Keep this as well
    )