import os
import io
import re
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path

streamlit_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('streamlit')]
for logger in streamlit_loggers:
    logger.setLevel(logging.ERROR)
import streamlit as st
import extra_streamlit_components as stx

from nnunetv2.evaluation.find_best_configuration import find_best_configuration
from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess, extract_fingerprints, plan_experiments
import nnunetv2.run.run_training as rt

from HD_BET.HD_BET.bpt import BrainPreProcessingTool
from ichilov2nnunet import ichilov_data_to_nnunet_format
from params import ToolKitParams, ToolKitStage



NNUNET_RAW_DATA_PATH = os.getenv("nnUNet_raw")
NNUNET_PREPROCESSED_PATH = os.getenv("nnUNet_preprocessed")
NNUNET_RESULTS = os.getenv("nnUNet_results")


class StreamlitLogger(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder

    def write(self, msg):
        super().write(msg)
        # Update the log display
        self.placeholder.text_area("Log Output", self.getvalue(), height=300)


def init_session_state():
    """Initialize session state variables only if they don't exist"""
    if 'initialized' not in st.session_state:
        # Main tabs
        st.session_state['active_tab'] = 'Data Preprocessing'

        # Preprocessing tab variables
        st.session_state['csv_path'] = None
        st.session_state['modalities'] = []
        st.session_state['bet_device'] = "0"
        st.session_state['modality_digits'] = {}
        st.session_state['task_id'] = None
        st.session_state['preprocessing_task_id_key'] = None
        st.session_state['training_task_id_key'] = None
        st.session_state['do_preprocessing'] = False
        st.session_state['perform_reg'] = False
        st.session_state['compress'] = True
        st.session_state['reg_fixed_modality'] = None
        st.session_state['perform_bet'] = False
        st.session_state['bet_save_out'] = True
        st.session_state['bet_out_path'] = ''
        st.session_state['labels'] = [{"label_number": 0, "label_string": "0"}]
        st.session_state['has_duplicate_mod_digits'] = False
        st.session_state['has_duplicate_label_digits'] = False

        # Training tab variables
        st.session_state['data_set'] = ""
        st.session_state['configuration'] = '3d_fullres'
        st.session_state['optimizer'] = "SGD"
        st.session_state['loss_function'] = "Dice CE"
        st.session_state['folds'] = []
        st.session_state['max_epochs'] = 50
        st.session_state['do_preprocessing_training'] = False
        st.session_state['do_transfer_lr'] = False

        # Mark as initialized
        st.session_state['initialized'] = True


def create_nnunet_dataset(params: ToolKitParams):
    with st.spinner("Creating nnUNet project..." if not task_folder_path(params.task) else "Appending to nnUNet project..."):
        ichilov_data_to_nnunet_format(params.csv_path, params, task_folder_path(params.task),
                                      not task_folder_exists(params.task))

        if not task_folder_exists(params.task):
            extract_fingerprints([int(params.task)])
            plans_identifier = plan_experiments([int(params.task)])
            preprocess([int(params.task)], plans_identifier, configurations=[params.configuration],
                       num_processes=[8])
        shutil.copyfile(
            os.path.join(NNUNET_RAW_DATA_PATH, f"Dataset{params.task:03d}", "splits_final.json"),
            os.path.join(NNUNET_PREPROCESSED_PATH, f"Dataset{params.task:03d}", "splits_final.json"))
    st.success("NNUnet Dataset is Ready")


# Callback functions for widgets
def update_perform_reg():
    st.session_state.perform_reg = st.session_state.preprocessing_perform_reg_key \
        if st.session_state.get('active_tab') == 'Data Preprocessing' \
        else st.session_state.inference_perform_reg_key


def update_compress():
    st.session_state.compress = st.session_state.compress_key


def update_reg_fixed_modality():
    st.session_state.reg_fixed_modality = st.session_state.preprocessing_reg_fixed_modality_key \
        if st.session_state.get('active_tab') == 'Data Preprocessing' \
        else st.session_state.inference_reg_fixed_modality_key

def update_perform_bet():
    st.session_state.perform_bet = st.session_state.preprocessing_perform_bet_key \
        if st.session_state.get('active_tab') == 'Data Preprocessing' \
        else st.session_state.inference_perform_bet_key

def update_bet_save_out():
    st.session_state.bet_save_out = st.session_state.bet_save_out_key


def update_bet_out_path():
    st.session_state.bet_out_path = st.session_state.preprocessing_bet_out_path_key \
        if st.session_state.get('active_tab') == 'Data Preprocessing' \
        else st.session_state.inference_bet_out_path_key

def update_do_preprocessing():
    st.session_state.do_preprocessing = st.session_state.preprocessing_do_preprocessing_key \
        if st.session_state.get('active_tab') == 'Data Preprocessing' \
        else st.session_state.inference_do_preprocessing_key



def update_csv_path():
    st.session_state.csv_path = st.session_state.csv_path_key


def update_modalities():
    st.session_state.modalities = st.session_state.modalities_key


def update_bet_device():
    st.session_state.bet_device = st.session_state.bet_device_key


def update_task_id():
    if st.session_state.active_tab == 'Data Preprocessing':
        st.session_state.task_id = st.session_state.preprocessing_task_id_key
    if st.session_state.active_tab == 'Training':
        st.session_state.task_id = st.session_state.training_task_id_key
    if st.session_state.active_tab == 'Inference':
        st.session_state.task_id = st.session_state.inference_task_id_key


def update_data_set():
    st.session_state.data_set = st.session_state.data_set_key


def update_configuration():
    st.session_state.configuration = st.session_state.configuration_key


def update_optimizer():
    st.session_state.optimizer = st.session_state.optimizer_key


def update_loss_function():
    st.session_state.loss_function = st.session_state.loss_function_key


def update_folds():
    st.session_state.folds = st.session_state.folds_key


def update_max_epochs():
    st.session_state.max_epochs = st.session_state.max_epochs_key


def update_do_preprocessing_training():
    st.session_state.do_preprocessing_training = st.session_state.do_preprocessing_training_key


def update_do_transfer_lr():
    st.session_state.do_transfer_lr = st.session_state.do_transfer_lr_key


def update_active_tab():
    if 'tab_key' in st.session_state:
        st.session_state.active_tab = st.session_state.tab_key


def _preprocessing_section(tab='preprocessing'):
    with st.expander("", expanded=True):
        st.header("Preprocessing")
        st.subheader("Registration Params")

        st.checkbox("Perform Registration",
                  value=st.session_state.get('perform_reg', False),
                  key=f"{tab}_perform_reg_key",
                  on_change=update_perform_reg)

        if st.session_state.get('perform_reg', True):
            with st.container():
                if st.session_state.get('reg_fixed_modality') is None and st.session_state.get('modalities', []):
                    st.session_state['reg_fixed_modality'] = st.session_state['modalities'][0]

                modalities = st.session_state.get('modalities', [])
                if modalities:
                    reg_fixed_modality = st.session_state.get('reg_fixed_modality')
                    default_index = modalities.index(reg_fixed_modality) if reg_fixed_modality in modalities else 0

                    st.selectbox(
                        "Registration Fixed Module",
                        options=modalities,
                        index=default_index,
                        key=f"{tab}_reg_fixed_modality_key",
                        on_change=update_reg_fixed_modality
                    )

        # Brain Extraction Section
        st.subheader("Brain Extraction Params")
        st.checkbox("Perform Brain Extraction",
                    value=st.session_state.get('perform_bet', True),
                    key=f"{tab}_perform_bet_key",
                    on_change=update_perform_bet)

        # Save Options Section
        st.subheader("Save Options")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("BET Output Path",
                          value=st.session_state.get('bet_out_path', ''),
                          key=f"{tab}_bet_out_path_key",
                          on_change=update_bet_out_path)


def _modalities_section(modalities_in=None, modality_digits_in=None):
    """
    Show the 'modalities' section.

    Parameters
    ----------
    modalities_in : list[str] | None
        If provided, overrides st.session_state['modalities'] for display.
    modality_digits_in : dict[str, str] | None
        If provided, overrides st.session_state['modality_digits'] for display.
        Values should be 4-digit strings like "0003".
    """
    st.write("modalities IDs:")
    read_only = (modalities_in is not None) and (modality_digits_in is not None)
    # ---- Source of truth for this render ----
    # If inputs are provided, use them strictly in read-only fashion (no writes).
    # Otherwise fall back to session_state (and allow edits/mutations when read_only=False).
    modalities = modalities_in if modalities_in is not None else st.session_state.get('modalities', [])
    digits = modality_digits_in if modality_digits_in is not None else st.session_state.get('modality_digits', {})

    # Helpers that mutate state — only wire them up if not read_only
    def update_modality_digits():
        """Assign incremental values to modality digits based on current modalities."""
        new_digits = {modality: f"{i:04}" for i, modality in enumerate(modalities)}
        st.session_state.modality_digits = new_digits

    def update_modality_digit(modality):
        digit = st.session_state.get(f"digit_{modality}", 0)
        current_digits = st.session_state.get('modality_digits', {})
        current_digits[modality] = f"{int(digit):04}"
        st.session_state.modality_digits = current_digits
        st.session_state.has_duplicate_mod_digits = (len(set(current_digits.values())) != len(current_digits))

    # In editable mode, keep session_state in sync and auto-fill missing digits
    if not read_only:
        if 'modalities' not in st.session_state and modalities_in is not None:
            st.session_state.modalities = modalities_in
        if 'modality_digits' not in st.session_state and modality_digits_in is not None:
            st.session_state.modality_digits = modality_digits_in

        # Make sure digits are aligned with modalities
        if len(st.session_state.get('modality_digits', {})) != len(modalities):
            update_modality_digits()

        # Re-read digits after possible update
        digits = st.session_state.get('modality_digits', {})

    # Layout
    if len(modalities) == 0:
        st.info("No modalities to display.")
        return

    cols = st.columns(len(modalities))

    for i, (modality, col) in enumerate(zip(modalities, cols)):
        # Determine the value to show
        default_value = i
        if modality in digits:
            try:
                default_value = int(digits[modality])
            except ValueError:
                default_value = i

        with col:
            st.write(f"{modality}")
            if read_only:
                # Display as a disabled number input with no callbacks and no state writes
                st.number_input(
                    f"Channel for {modality}",
                    min_value=0,
                    max_value=max(len(modalities) - 1, 0),
                    value=default_value,
                    key=f"ro_digit_{modality}",  # separate key to avoid colliding with editable state
                    label_visibility="collapsed",
                    disabled=True,
                )
            else:
                st.number_input(
                    f"Channel for {modality}",
                    min_value=0,
                    max_value=max(len(modalities) - 1, 0),
                    value=default_value,
                    key=f"digit_{modality}",
                    on_change=update_modality_digit,
                    args=(modality,),
                    label_visibility="collapsed",
                    disabled=False,
                )

    # Only warn about duplicates in editable mode
    if not read_only and st.session_state.get('has_duplicate_mod_digits', False):
        with st.container():
            st.warning("Please assign unique digits to each modality.")


def _labels_section(labels_in=None):
    """
    Show the 'labels' section.

    Parameters
    ----------
    labels_in : list[dict] | None
        If provided, overrides st.session_state['labels'] for display.
        Each item: {"label_number": int, "label_string": str}
        NOTE: Index 0 is expected to be the background label (0, "background").
    """
    st.write("Labels")
    read_only = labels_in is not None
    # When editable, we may normalize/enforce background label.
    # When read_only, we must NOT mutate input or session state.
    def ensure_background_label():
        labels = st.session_state.get('labels', [])
        if not labels or labels[0].get("label_number") != 0:
            background_label = {"label_number": 0, "label_string": "background"}
            if labels and labels[0].get("label_number") != 0:
                labels.insert(0, background_label)
            elif not labels:
                labels = [background_label]
            st.session_state.labels = labels
        else:
            labels[0]["label_number"] = 0
            labels[0]["label_string"] = "background"
        return st.session_state.labels

    def add_pair():
        labels = ensure_background_label()
        used_numbers = {label["label_number"] for label in labels}
        next_num = 1
        while next_num in used_numbers:
            next_num += 1
        labels.append({"label_number": next_num, "label_string": str(next_num)})
        st.session_state.labels = labels

    def delete_pair(index):
        labels = st.session_state.get('labels', [])
        if index > 0 and len(labels) > 2:
            labels.pop(index)
            st.session_state.labels = labels

    def update_label_number(i):
        number = st.session_state[f"num_{i}"]
        labels = st.session_state.get('labels', [])

        if i > 0 and number == 0:
            st.error("Label number 0 is reserved for background. Please choose a different number.")
            return

        if i < len(labels):
            labels[i]["label_number"] = number
            if i > 0:
                labels[i]["label_string"] = str(number)
            st.session_state.labels = labels

        st.session_state.has_duplicate_label_digits = (len({l["label_number"] for l in labels}) != len(labels))

    def update_label_string(i):
        string = st.session_state[f"str_{i}"]
        labels = st.session_state.get('labels', [])
        if i < len(labels):
            if i == 0:
                labels[i]["label_string"] = "background"
            else:
                labels[i]["label_string"] = string
            st.session_state.labels = labels

    # Determine labels to display
    if read_only:
        labels = labels_in if labels_in is not None else st.session_state.get('labels', [])
        # Do not mutate. If missing a background row, just display as-is.
    else:
        # Editable mode: initialize session state from input if provided
        if labels_in is not None and 'labels' not in st.session_state:
            st.session_state.labels = labels_in

        # Ensure background is present/correct when editing
        labels = ensure_background_label()

    if not labels:
        st.info("No labels to display.")
        return

    for i, pair in enumerate(labels):
        cols = st.columns([3, 5, 1])
        is_background_label = (i == 0)

        with cols[0]:
            if read_only:
                st.number_input(
                    f"Label {i + 1}",
                    value=pair.get("label_number", 0),
                    key=f"ro_num_{i}",
                    label_visibility="collapsed",
                    disabled=True,
                    min_value=0 if is_background_label else 1
                )
            else:
                st.number_input(
                    f"Label {i + 1}",
                    value=pair.get("label_number", 0),
                    key=f"num_{i}",
                    on_change=update_label_number,
                    args=(i,),
                    label_visibility="collapsed",
                    disabled=is_background_label,
                    min_value=0 if is_background_label else 1
                )

        with cols[1]:
            if read_only:
                st.text_input(
                    f"Label Name {i + 1}",
                    value=pair.get("label_string", ""),
                    key=f"ro_str_{i}",
                    label_visibility="collapsed",
                    disabled=True,
                )
            else:
                st.text_input(
                    f"Label Name {i + 1}",
                    value=pair.get("label_string", ""),
                    key=f"str_{i}",
                    on_change=update_label_string,
                    args=(i,),
                    label_visibility="collapsed",
                    disabled=is_background_label
                )

        with cols[2]:
            if read_only:
                st.write("🔒" if is_background_label else "—")
            else:
                if not is_background_label and len(labels) > 2:
                    st.button("❌", key=f"del_{i}", on_click=delete_pair, args=(i,))
                elif is_background_label:
                    st.write("🔒")

    # Add button & duplicate warning only in editable mode
    if not read_only:
        st.button("Add Another Label", key="add_label_btn", on_click=add_pair)

        if st.session_state.get('has_duplicate_label_digits', False):
            with st.container():
                st.warning("Please assign unique digits to each label.")


def post_training(params: ToolKitParams):
    find_best_configuration(params.task,
                            tuple([{'plans': 'nnUNetPlans', 'configuration': params.configuration, 'trainer': 'nnUNetTrainer'},]),
                            True, 8, False, params.folds)


def find_first_postprocessing_pkl(root_dir: str) -> str | None:
    match = next(Path(root_dir).rglob("postprocessing.pkl"), None)
    return str(match) if match else None


def post_inference(params: ToolKitParams):
    predictTs_dir = os.path.join(task_folder_path(params.task), "predictTs")
    post_predictTs_dir = os.path.join(task_folder_path(params.task), "predictTs", "post_processed")
    results_dir = task_folder_path(params.task, NNUNET_RESULTS)
    subprocess.run([
        "nnUNetv2_apply_postprocessing",
        "-i", predictTs_dir,
        "-o", post_predictTs_dir,
        "-pp_pkl_file", find_first_postprocessing_pkl(results_dir),
    ])


def task_folder_path(task_id, base_folder=NNUNET_RAW_DATA_PATH):
    return os.path.join(base_folder, f"Dataset{task_id:03d}")


def task_folder_exists(task_id):
    return os.path.exists(task_folder_path(task_id))


def run_training(params: ToolKitParams):
    num_folds = len(params.folds)
    progress_bar = st.progress(0.0)

    with st.spinner("Training..."):
        log_placeholder = st.empty()
        # logger = StreamlitLogger(log_placeholder)
        # sys.stdout = logger
        log_container = st.empty()

        for idx, f in enumerate(params.folds):
            # Calculate progress as a float between 0 and 1
            progress = (idx) / num_folds
            progress_bar.progress(progress)
            # Display current fold
            st.write(f"Processing fold {f} ({idx + 1} of {num_folds})")


            # Prepare parameters for training
            Params = {
                "Optimizer": params.optimizer,
                "LossFunction": params.loss_function,
                "Epochs": int(params.max_epochs),
                "Transfer": params.do_transfer_lr,
                "DataSet": str(params.task),
                "Configuration": params.configuration,
                "Fold": f,
                "log_container": log_container
            }
            # Run training
            rt.main(Params)
        progress_bar.progress(1.0)
        st.success(f"Done Training")
        post_training(params)

def run_inference(params: ToolKitParams) -> str:
    progress_bar = st.progress(0)

    imagesTs_dir = os.path.join(task_folder_path(params.task), "imagesTs")
    predictTs_dir = os.path.join(task_folder_path(params.task), "predictTs")
    os.makedirs(predictTs_dir, exist_ok=True)

    prefixes = sorted({re.match(r"(NNUNET_\d{3})", f).group(1)
                       for f in os.listdir(imagesTs_dir)
                       if re.match(r"(NNUNET_\d{3})", f)})

    for i, prefix in enumerate(prefixes):
        # create temp folder for this case
        progress_bar.progress(i/len(prefixes))
        tmp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")

        # copy all modality files for this case
        for fname in os.listdir(imagesTs_dir):
            if fname.startswith(prefix):
                shutil.copy(os.path.join(imagesTs_dir, fname), os.path.join(tmp_dir, fname))

        subprocess.run([
            "nnUNetv2_predict",
            "-i", tmp_dir,
            "-o", predictTs_dir,
            "-d", f"Dataset{params.task:03d}",
            "-c", params.configuration
        ])

    post_inference(params)
    progress_bar.progress(1.0)
    st.success(f"Done Inference")


def run_preprocessing(params: ToolKitParams) -> str:
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_container = st.container()
    processor = BrainPreProcessingTool(params)
    errors = []
    last_output_directory = ""

    with st.spinner("Processing..."):

        for info in processor.preprocess():
            if "progress" in info:
                progress_value = float(info["progress"].strip("%")) / 100
                progress_bar.progress(progress_value)
                status_text.text(f"Processing {info['subject_name']} - {info['study']}")
            elif info.get("status") == "failed":
                # Collect errors
                errors.append(f"{info['subject_name']}: {info.get('error', 'Unknown error')}")
                # Show in error container in real time
                with error_container:
                    st.error(f"Error processing {info['subject_name']}: {info.get('error', 'Unknown error')}")
            elif info.get("status") == "finished":
                progress_bar.progress(1.00)
                last_output_directory = info.get('last_output_directory', '')

        # Final summary
        if errors:
            st.warning(f"Completed with {len(errors)} errors")
        else:
            st.success("Processing completed successfully!")

        return last_output_directory


def show_gui():
    st.header("nnUNet Toolkit")
    init_session_state()

    # Use extra_streamlit_components tab component to track the active tab
    tab_options = ['Data Preprocessing', 'Training', 'Inference']
    chosen_tab = stx.tab_bar(
        data=[stx.TabBarItemData(id=tab, title=tab, description="") for tab in tab_options],
        default=st.session_state.get('active_tab', 'Data Preprocessing'),
        key="tab_key"
    )

    # Update session state with current tab
    if chosen_tab and chosen_tab != st.session_state.get('active_tab'):
        st.session_state['active_tab'] = chosen_tab
        update_active_tab()

    # Display content based on active tab
    if st.session_state.get('active_tab') == 'Data Preprocessing':
        with st.container(border=True):
            st.subheader("Primary Parameters")

            st.file_uploader(
                "CSV path",
                type=["csv"],
                key="csv_path_key",
                on_change=update_csv_path
            )

            st.text_input(
                "BET Device (ID or 'CPU')",
                value=st.session_state.get('bet_device', "0"),
                key="bet_device_key",
                on_change=update_bet_device
            )

            with st.container():
                st.subheader("nnUNet Parameters")
                task_id = st.number_input(
                    label="Output nnUNet Task",
                    min_value=1, step=1,
                    value=st.session_state.get('task_id', None),
                    placeholder="task number (E.g. 2)",
                    help="Enter the task number",
                    key="preprocessing_task_id_key",
                    on_change=update_task_id
                )
                labels_in = None
                modalities_in = None
                modality_digits_in = None
                if task_id is not None and task_folder_exists(task_id):
                    params = ToolKitParams.load_from_json(task_folder_path(task_id))
                    labels_in = [{"label_number": v, "label_string": k} for k,v in params.label_ids.items()]
                    modality_digits_in = params.modality_ids
                    modalities_in = params.modalities
                    st.session_state.modalities = modalities_in
                    st.session_state.modality_digits = modality_digits_in

                st.multiselect(
                    "Modalities",
                    options=['T1', 'T2', 'T1ce', 'FLAIR'],
                    default=st.session_state.get('modalities', []),
                    key="modalities_key",
                    on_change=update_modalities,
                    disabled=modalities_in is not None
                )
                if st.session_state.get('modalities', []):
                    _modalities_section(modalities_in, modality_digits_in)
                    _labels_section(labels_in)

        st.checkbox(
            "Perform Preprocessing",
            value=st.session_state.get('do_preprocessing', False),
            key="preprocessing_do_preprocessing_key",
            on_change=update_do_preprocessing
        )

        if st.session_state.get('do_preprocessing', False):
            _preprocessing_section()

    elif st.session_state.get('active_tab') == 'Training':
        with st.expander("", expanded=True):
            st.subheader("nnUNet Training Configuration")
            choices_loss_function = [
                "Dice CE",
                "Robust Cross-Entropy Loss",
                "Dice Focal",
                "Dice Top-K 10",
                "Dice Top-K 10 CE",
                "Dice Top-K 10 Focal"
            ]
            choices_optimizer = ["Adam", "SGD"]
            choices_folds = ['0', '1', '2', '3', '4']

            # Dataset and Configuration
            col1, col2 = st.columns(2)
            with col1:
                st.number_input(
                    label="Output nnUNet Task",
                    min_value=1, step=1,
                    value=st.session_state.get('task_id', None),
                    placeholder="task number (E.g. 2)",
                    help="Enter the task number",
                    key="training_task_id_key",
                    on_change=update_task_id
                )

            with col2:
                configuration_options = ['3d_fullres', '3d_cascade_fullres', '3d_lowers', '2d']
                current_config = st.session_state.get('configuration', '3d_fullres')
                config_index = configuration_options.index(
                    current_config) if current_config in configuration_options else 0

                st.selectbox(
                    "Configuration",
                    options=configuration_options,
                    index=config_index,
                    key="configuration_key",
                    on_change=update_configuration
                )

            # Optimizer and Loss Function
            col1, col2 = st.columns(2)
            with col1:
                optimizer_options = choices_optimizer
                current_optimizer = st.session_state.get('optimizer', 'SGD')
                optimizer_index = optimizer_options.index(
                    current_optimizer) if current_optimizer in optimizer_options else 0

                st.radio(
                    "Optimizer",
                    options=optimizer_options,
                    index=optimizer_index,
                    key="optimizer_key",
                    on_change=update_optimizer
                )

            with col2:
                loss_function_options = choices_loss_function
                current_loss = st.session_state.get('loss_function', 'Dice CE')
                loss_index = loss_function_options.index(current_loss) if current_loss in loss_function_options else 0

                st.radio(
                    "Loss Function",
                    options=loss_function_options,
                    index=loss_index,
                    key="loss_function_key",
                    on_change=update_loss_function
                )

            # Folds Selection
            st.multiselect(
                "Select Folds",
                options=choices_folds,
                default=st.session_state.get('folds', []),
                key="folds_key",
                on_change=update_folds
            )

            # Epochs and Processing Options
            col1, col2 = st.columns(2)
            with col1:
                st.number_input(
                    "Maximum Epochs",
                    min_value=1,
                    value=st.session_state.get('max_epochs', 50),
                    help="Maximum number of training epochs",
                    key="max_epochs_key",
                    on_change=update_max_epochs
                )

    elif st.session_state.get('active_tab') == 'Inference':
        with st.expander("", expanded=True):
            st.subheader("nnUNet Inference")

            col1, col2 = st.columns(2)
            with col1:
                st.number_input(
                    label="Output nnUNet Task",
                    min_value=1, step=1,
                    value=st.session_state.get('task_id', None),
                    placeholder="task number (E.g. 2)",
                    help="Enter the task number",
                    key="inference_task_id_key",
                    on_change=update_task_id
                )


    # Run button based on active tab
    can_run = False
    if st.session_state.get('active_tab') == 'Data Preprocessing':
        can_run = not (st.session_state.has_duplicate_mod_digits or st.session_state.has_duplicate_label_digits) \
                  and (len(st.session_state.modalities)) and st.session_state.task_id != "" \
                  and st.session_state.csv_path is not None
    if st.session_state.get('active_tab') == 'Training':
        can_run = st.session_state.task_id is not None and task_folder_exists(st.session_state.task_id) \
                  and len(st.session_state.folds)
    if st.session_state.get('active_tab') == 'Inference':
        can_run = st.session_state.task_id is not None and task_folder_exists(st.session_state.task_id)
    if st.button(f"Run {st.session_state.get('active_tab', 'Data Preprocessing')}", disabled=not can_run):
        if st.session_state.get('active_tab') == 'Data Preprocessing':
            # Preprocessing action
            try:
                label_ids = {str(l['label_string']): int(l['label_number']) for l in st.session_state.get('labels', {})}
                params = ToolKitParams(
                    task=int(st.session_state.get('task_id', 0)),
                    csv_path=st.session_state.get('csv_path'),
                    out_path=st.session_state.get('bet_out_path', ''),
                    reg_fixed_module=st.session_state.get('reg_fixed_modality') if st.session_state.get('perform_reg') else None,
                    shrink_output=st.session_state.get('compress', True),
                    modalities=st.session_state.get('modalities', []),
                    modality_ids=st.session_state.get('modality_digits', {}),
                    perform_reg=st.session_state.perform_reg,
                    perform_bet=st.session_state.perform_bet,
                    label_ids=label_ids
                )

                if st.session_state.get('do_preprocessing', False):
                    last_output_directory = run_preprocessing(params)
                    params.csv_path = os.path.join(last_output_directory, 'summary.csv')

                create_nnunet_dataset(params)
                ToolKitParams.update(os.path.join(NNUNET_RAW_DATA_PATH, f"Dataset{params.task:03d}"),
                                     params, ToolKitStage.PREPROCESSING)
            except ValueError as e:
                st.error(f"Error: {e}")
                st.error("Please check your input parameters.")

        elif st.session_state.get('active_tab') == 'Training':
            params = ToolKitParams(
                task=int(st.session_state.get('task_id', 0)),
                optimizer=st.session_state.get('optimizer', ''),
                loss_function=st.session_state.get('loss_function', ''),
                configuration=st.session_state.get('configuration', ''),
                folds=st.session_state.get('folds', []),
                max_epochs=st.session_state.get('max_epochs', 50),
                do_preprocessing_training=st.session_state.get('do_preprocessing_training', True),
                do_transfer_lr=st.session_state.get('do_transfer_lr', False)
            )
            run_training(params)
            ToolKitParams.update(
                os.path.join(NNUNET_RAW_DATA_PATH, f"Dataset{params.task:03d}"),
                params, ToolKitStage.TRAINING)

        elif st.session_state.get('active_tab') == 'Inference':
            st.write("Starting inference with the configured parameters...")
            params = ToolKitParams.load_from_json(task_folder_path(st.session_state.get('task_id', 0)))
            run_inference(params)


if __name__ == "__main__":
    st.set_page_config(page_title="nnUNetToolkit", layout="wide")
    show_gui()
