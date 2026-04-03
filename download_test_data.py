
from pathlib import Path
import os
import warnings
import tqdm
import requests
import netrc

from nirvana.util.download import download_file
from nirvana.tests.util import remote_data_file, remote_drp_test_files, remote_drp_test_images
from nirvana.tests.util import remote_dap_test_files
from nirvana.tests.util import drp_test_version, dap_test_version, dap_test_daptype


def main():

    from IPython import embed

    local_root = Path(remote_data_file()).resolve()
    if not local_root.exists():
        local_root.mkdir(parents=True)

    dr = 'DR17'
    url_root = f'https://data.sdss.org/sas/{dr.lower()}/manga/spectro'

    # DRP files
    drp_files = remote_drp_test_files()
    drp_images = remote_drp_test_images()
    plates = [f.split('-')[1] for f in drp_files]
    for plate, fcube, fimg in zip(plates, drp_files, drp_images):
        local_file = local_root / fcube
        if local_file.exists():
            warnings.warn(f'{local_file.name} exists.  Skipping...')
        else:
            url = f'{url_root}/redux/{drp_test_version}/{plate}/stack/{fcube}'
            download_file(url, local_file)

        local_file = local_root / fimg
        if local_file.exists():
            warnings.warn(f'{local_file.name} exists.  Skipping...')
        else:
            url = f'{url_root}/redux/{drp_test_version}/{plate}/images/{fimg}'
            download_file(url, local_file)

    # DAP files
    dap_files = remote_dap_test_files(daptype=dap_test_daptype)
    plates = [f.split('-')[1] for f in dap_files]
    ifus = [f.split('-')[2] for f in dap_files]
    for plate, ifu, f in zip(plates, ifus, dap_files):
        local_file = local_root / f
        if local_file.exists():
            warnings.warn(f'{local_file.name} exists.  Skipping...')
            continue

        url = f'{url_root}/analysis/{drp_test_version}/{dap_test_version}/{dap_test_daptype}' \
              f'/{plate}/{ifu}/{f}'
        download_file(url, local_file)

    # DRPall file
    local_file = local_root / f'drpall-{drp_test_version}.fits'
    if local_file.exists():
        warnings.warn(f'{local_file.name} exists.  Skipping...')
    else:
        url = f'{url_root}/redux/{drp_test_version}/{local_file.name}'
        download_file(url, local_file)

    # DAPall file
    local_file = local_root / f'dapall-{drp_test_version}-{dap_test_version}.fits'
    if local_file.exists():
        warnings.warn(f'{local_file.name} exists.  Skipping...')
    else:
        url = f'{url_root}/analysis/{drp_test_version}/{dap_test_version}/{local_file.name}'
        download_file(url, local_file)


if __name__ == '__main__':
    main()


