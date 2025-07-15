import os
import sys
import io
import shutil
import streamlit as st
import logging

streamlit_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('streamlit')]
for logger in streamlit_loggers:
    logger.setLevel(logging.ERROR)

import streamlit as st
import extra_streamlit_components as stx
from HD_BET.HD_BET.bpt import BrainPreProcessingTool
from ichilov2nnunet import ichilov_data_to_nnunet_format
from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess, extract_fingerprints, plan_experiments
from params import ToolKitParams
import nnunetv2.run.run_training as rt



NNUNET_RAW_DATA_PATH = os.getenv("nnUNet_raw")
NNUNET_PREPROCESSED_PATH = os.getenv("nnUNet_preprocessed")

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
        st.session_state['task_id'] = ""
        st.session_state['preprocessing_task_id_key'] = ""
        st.session_state['training_task_id_key'] = ""
        st.session_state['do_preprocessing'] = False
        st.session_state['perform_reg'] = True
        st.session_state['compress'] = True
        st.session_state['reg_fixed_modality'] = None
        st.session_state['perform_bet'] = True
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
    out_path = os.path.join(NNUNET_RAW_DATA_PATH, f"Dataset{params.task:03d}")
    with st.spinner("Creating nnUNet project..."):
        ichilov_data_to_nnunet_format(params.csv_path, params, out_path)

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
    st.session_state.perform_reg = st.session_state.perform_reg_key


def update_compress():
    st.session_state.compress = st.session_state.compress_key


def update_reg_fixed_modality():
    st.session_state.reg_fixed_modality = st.session_state.reg_fixed_modality_key


def update_perform_bet():
    st.session_state.perform_bet = st.session_state.perform_bet_key


def update_bet_save_out():
    st.session_state.bet_save_out = st.session_state.bet_save_out_key


def update_bet_out_path():
    st.session_state.bet_out_path = st.session_state.bet_out_path_key


def update_do_preprocessing():
    st.session_state.do_preprocessing = st.session_state.do_preprocessing_key


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


def _preprocessing_section():
    with st.expander("", expanded=True):
        st.header("Preprocessing")
        st.subheader("Registration Params")

        perform_reg = st.checkbox("Perform Registration",
                                  value=st.session_state.get('perform_reg', True),
                                  key="perform_reg_key",
                                  on_change=update_perform_reg)

        if st.session_state.get('perform_reg', True):
            with st.container():
                # compress = st.checkbox('Compress (nii.gz)',
                #                        value=st.session_state.get('compress', True),
                #                        key="compress_key",
                #                        on_change=update_compress)

                # Set default reg_fixed_modality if not set and modalities exist
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
                        key="reg_fixed_modality_key",
                        on_change=update_reg_fixed_modality
                    )

        # Brain Extraction Section
        st.subheader("Brain Extraction Params")
        st.checkbox("Perform Brain Extraction",
                    value=st.session_state.get('perform_bet', True),
                    key="perform_bet_key",
                    on_change=update_perform_bet)

        # Save Options Section
        st.subheader("Save Options")
        col1, col2 = st.columns(2)
        with col1:
            # perform_bet = st.session_state.get('perform_bet', True)
            # st.checkbox("Save BET Output",
            #             value=st.session_state.get('bet_save_out', True),
            #             disabled=not perform_bet,
            #             key="bet_save_out_key",
            #             on_change=update_bet_save_out)
        #
        # with col2:
            st.text_input("BET Output Path",
                          value=st.session_state.get('bet_out_path', ''),
                          key="bet_out_path_key",
                          on_change=update_bet_out_path)


def _modalities_section():
    st.write("modalities IDs:")
    modalities = st.session_state.get('modalities', [])
    digits = st.session_state.get('modality_digits', {})

    def update_modality_digits():
        """Assign incremental values to modality digits based on current modalities."""
        new_digits = {}

        # Rearrange digits incrementally
        for i, modality in enumerate(modalities):
            new_digits[modality] = f"{i:04}"
        st.session_state.modality_digits = new_digits

    def update_modality_digit(modality):
        digit = st.session_state[f"digit_{modality}"]
        digits = st.session_state.get('modality_digits', {})
        digits[modality] = f"{digit:04}"
        st.session_state.modality_digits = digits

        if len(set(digits.values())) != len(digits):
            st.session_state.has_duplicate_mod_digits = True
        else:
            st.session_state.has_duplicate_mod_digits = False

    cols = st.columns(len(modalities))

    if len(digits) != len(modalities):
        update_modality_digits()

    for i, (modality, col) in enumerate(zip(modalities, cols)):
        default_value = i
        digits = st.session_state.get('modality_digits', {})
        if modality in digits:
            try:
                default_value = int(digits[modality])
            except ValueError:
                default_value = i

        with col:
            st.write(f"{modality}")
            st.number_input(
                f"Channel for {modality}",
                min_value=0,
                max_value=len(modalities) - 1,
                value=default_value,
                key=f"digit_{modality}",
                on_change=update_modality_digit,
                args=(modality,),
                label_visibility="collapsed"
            )

    if st.session_state.get('has_duplicate_mod_digits', False):
        with st.container():
            st.warning("Please assign unique digits to each modality.")


def _labels_section():
    st.write("Labels")

    # Ensure background label (0) always exists and is first
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
            # Ensure the first label always has correct values
            labels[0]["label_number"] = 0
            labels[0]["label_string"] = "background"
        return st.session_state.labels

    def add_pair():
        labels = ensure_background_label()
        # Find next available number (excluding 0 which is reserved)
        used_numbers = {label["label_number"] for label in labels}
        next_num = 1
        while next_num in used_numbers:
            next_num += 1
        labels.append({"label_number": next_num, "label_string": str(next_num)})
        st.session_state.labels = labels

    def delete_pair(index):
        labels = st.session_state.get('labels', [])
        # Never delete the background label (index 0) and ensure we have at least 2 labels
        if index > 0 and len(labels) > 2:
            labels.pop(index)
            st.session_state.labels = labels

    def update_label_number(i):
        number = st.session_state[f"num_{i}"]
        labels = st.session_state.get('labels', [])

        # Prevent setting any label other than index 0 to number 0
        if i > 0 and number == 0:
            st.error("Label number 0 is reserved for background. Please choose a different number.")
            return

        if i < len(labels):
            labels[i]["label_number"] = number
            # Don't auto-update string for background label
            if i > 0:
                labels[i]["label_string"] = str(number)
            st.session_state.labels = labels

        # Check for duplicates
        if len(set([l["label_number"] for l in labels])) != len(labels):
            st.session_state.has_duplicate_label_digits = True
        else:
            st.session_state.has_duplicate_label_digits = False

    def update_label_string(i):
        string = st.session_state[f"str_{i}"]
        labels = st.session_state.get('labels', [])
        if i < len(labels):
            # Don't allow changing the background label string
            if i == 0:
                labels[i]["label_string"] = "background"
            else:
                labels[i]["label_string"] = string
            st.session_state.labels = labels

    # Ensure background label exists
    labels = ensure_background_label()

    for i, pair in enumerate(labels):
        cols = st.columns([3, 5, 1])

        is_background_label = (i == 0)

        with cols[0]:
            st.number_input(
                f"Label {i + 1}",
                value=pair.get("label_number", 0),
                key=f"num_{i}",
                on_change=update_label_number,
                args=(i,),
                label_visibility="collapsed",
                disabled=is_background_label,
                min_value=0 if is_background_label else 1  # Background can be 0, others start at 1
            )

        with cols[1]:
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
            if not is_background_label and len(labels) > 2:
                st.button("❌", key=f"del_{i}", on_click=delete_pair, args=(i,))
            elif is_background_label:
                st.write("🔒")  # Show lock icon for protected background label

    st.button("Add Another Label", key="add_label_btn", on_click=add_pair)

    if st.session_state.get('has_duplicate_label_digits', False):
        with st.container():
            st.warning("Please assign unique digits to each label.")


def run_training(params: ToolKitParams):
    num_folds = len(params.folds)
    progress_bar = st.progress(0.0)

    with st.spinner("Training..."):
        log_placeholder = st.empty()
        # logger = StreamlitLogger(log_placeholder)
        # sys.stdout = logger
        for idx, f in enumerate(params.folds):
            # Calculate progress as a float between 0 and 1
            progress = (idx) / num_folds

            # Display current fold
            st.write(f"Processing fold {f} ({idx + 1} of {num_folds})")

            log_container = st.empty()

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
            trainer = rt.main(Params)
        progress_bar.progress(1)
        st.success(f"Done Training")


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

            uploaded_file = st.file_uploader(
                "CSV path",
                type=["csv"],
                key="csv_path_key",
                on_change=update_csv_path
            )

            st.multiselect(
                "Modalities",
                options=['T1', 'T2', 'T1ce', 'FLAIR'],
                default=st.session_state.get('modalities', []),
                key="modalities_key",
                on_change=update_modalities
            )

            st.text_input(
                "BET Device (ID or 'CPU')",
                value=st.session_state.get('bet_device', "0"),
                key="bet_device_key",
                on_change=update_bet_device
            )

            with st.container():
                st.subheader("nnUNet Parameters")
                if st.session_state.get('modalities', []):
                    _modalities_section()
                    _labels_section()

                st.text_input(
                    "Output nnUNet Task",
                    value=st.session_state.get('task_id', ""),
                    placeholder="task number (E.g. 2)",
                    help="Enter the task number",
                    key="preprocessing_task_id_key",
                    on_change=update_task_id
                )

        st.checkbox(
            "Perform Preprocessing",
            value=st.session_state.get('do_preprocessing', False),
            key="do_preprocessing_key",
            on_change=update_do_preprocessing
        )

        if st.session_state.get('do_preprocessing', False):
            _preprocessing_section()

    elif st.session_state.get('active_tab') == 'Training':
        with st.expander("", expanded=True):
            st.header("nnUNet Training Configuration")
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
                st.text_input(
                    "Input nnUNet Task",
                    value=st.session_state.get('task_id', ""),
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

            # Checkboxes for preprocessing and transfer learning
            # with col2:
            #     st.checkbox(
            #         "Enable Preprocessing",
            #         value=st.session_state.get('do_preprocessing_training', False),
            #         help="Enable data preprocessing steps",
            #         key="do_preprocessing_training_key",
            #         on_change=update_do_preprocessing_training
            #     )
            #
            #     st.checkbox(
            #         "Enable Transfer Learning",
            #         value=st.session_state.get('do_transfer_lr', False),
            #         help="Enable transfer learning",
            #         key="do_transfer_lr_key",
            #         on_change=update_do_transfer_lr
            #     )

    elif st.session_state.get('active_tab') == 'Inference':
        st.subheader("Inference")
        st.info("Inference configuration will be added here.")
        # Your inference tab content here - add session state variables when implementing

    # Run button based on active tab
    can_run = False
    if st.session_state.get('active_tab') == 'Data Preprocessing':
        can_run = not (st.session_state.has_duplicate_mod_digits or st.session_state.has_duplicate_label_digits) \
                  and (len(st.session_state.modalities)) and st.session_state.task_id != "" \
                  and st.session_state.csv_path is not None
    if st.session_state.get('active_tab') == 'Training':
        can_run = st.session_state.task_id != ""  and len(st.session_state.folds)
    if st.button(f"Run {st.session_state.get('active_tab', 'Data Preprocessing')}", disabled=not can_run):
        if st.session_state.get('active_tab') == 'Data Preprocessing':
            # Preprocessing action
            try:
                label_ids = {str(l['label_string']): int(l['label_number']) for l in st.session_state.get('labels', {})}
                params = ToolKitParams(
                    task=int(st.session_state.get('task_id', 0)),
                    csv_path=st.session_state.get('csv_path'),
                    out_path=st.session_state.get('bet_out_path', ''),
                    reg_fixed_module=st.session_state.get('reg_fixed_modality'),
                    shrink_output=st.session_state.get('compress', True),
                    modalities=st.session_state.get('modalities', []),
                    modality_ids=st.session_state.get('modality_digits', {}),
                    perform_bet=st.session_state.perform_bet,
                    label_ids=label_ids
                )

                if st.session_state.get('do_preprocessing', False):
                    last_output_directory = run_preprocessing(params)
                    params.csv_path = os.path.join(last_output_directory, 'summary.csv')

                create_nnunet_dataset(params)

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
            st.write("Starting training with the configured parameters...")
            # Add your training function here

        elif st.session_state.get('active_tab') == 'Inference':
            # Inference action
            st.write("Starting inference with the configured parameters...")
            # Add your inference function here


if __name__ == "__main__":
    st.set_page_config(page_title="nnUNetToolkit", layout="wide")
    show_gui()
