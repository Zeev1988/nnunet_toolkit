import shutil
import os
import subprocess
import logging
import SimpleITK as sitk


def register(subject, exe_path, param_path, output_dir, modalities, fixed_module, label, shrink):

    study = subject[1]

    logging.info("Registration - started")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    modules = list(study.values())[0]
    fixed_path = modules[fixed_module]

    if fixed_path is None or not os.path.exists(fixed_path):
        logging.error(f"Registration - registration file was not found {fixed_path}")
        print(f"Registration - Error: registration file was not found {fixed_path}")
        logging.error("Registration - failed - exiting")
        print("Registration - failed - exiting")
        return

    if not fixed_path.endswith('.nii') and not fixed_path.endswith('.nii.gz'):
        logging.error(f"Registration - registration file is not of type nifty {fixed_path}")
        print(f"Registration - registration file is not of type nifty {fixed_path}")
        logging.error("Registration - failed - exiting")
        print("Registration - failed - exiting")
        return

    temp_dir = os.path.join(output_dir, r'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_output = os.path.join(temp_dir, 'result.0.nii.gz')

    print(f'\nRunning image registration\n')
    for modality in modalities + ['Label']:
        if modules[modality]:
            if not os.path.exists(modules[modality]):
                logging.error(f"Registration - Could not find {modules[modality]}")
                print(f"Registration - Error: could not find {modules[modality]}")
                continue

            if not modules[modality].endswith('.nii') and not modules[modality].endswith('.nii.gz'):
                logging.error(f"Registration - file is not of type nifty {modules[modality]}")
                print(f"Registration - Error: file is not of type nifty {modules[modality]}")
                continue

            m_path = os.path.join(output_dir, os.path.basename(modules[modality]))
            if modality == fixed_module:
                if shrink and modules[modality].endswith('.nii'):
                    m_path += ".gz"
                    image = sitk.ReadImage(modules[modality])
                    sitk.WriteImage(image, m_path)
                    os.remove(modules[modality])
                    if modality == fixed_module:
                        fixed_path = m_path
                elif modules[modality] != m_path:
                    shutil.copyfile(modules[modality], m_path)
            else:
                extract = False
                if modules[modality].endswith('.nii'):
                    extract = not shrink
                    m_path += ".gz"
                ret = subprocess.run([str(exe_path), '-f', str(fixed_path), '-m', str(modules[modality]),
                                      '-out', str(temp_dir), '-p', str(param_path)], capture_output=True)

                logging.info(ret.args)
                if ret.returncode:
                    logging.error(ret.stderr)
                    print("Registration - " + ret.stderr.decode("utf-8"))
                else:
                    logging.info(ret.stdout)

                if os.path.exists(temp_output):
                    if extract:
                        m_path = m_path.replace("nii.gz", "nii")
                        image = sitk.ReadImage(modules[modality])
                        sitk.WriteImage(image, m_path)
                    else:
                        shutil.copyfile(temp_output, m_path)

                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))

            modules[modality] = m_path
    os.rmdir(temp_dir)

    logging.info("Registration - ended")
