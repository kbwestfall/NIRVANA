"""
Script to download MaNGA data needed to run nirvana.
"""
import os
import argparse

from IPython import embed

from ..data import manga
from . import scriptbase


class MaNGACatalogs(scriptbase.ScriptBase):

    @classmethod
    def get_parser(cls, width=None):
        parser = super().get_parser(
            description='Download MaNGA survey catalogs', width=width, default_log_file=True
        )

        parser.add_argument('--dr', default='MPL-11', type=str, help='The MaNGA data release.')
        parser.add_argument(
            '--redux', default=None, type=str,
            help=(
                'Top-level directory with the MaNGA DRP output. This is used to construct the '
                'output directory structure to mimic the SAS. If not defined and the direct root '
                'for the output is also not defined (see --root), this is set by the '
                'environmental variable MANGA_SPECTRO_REDUX.'
            )
        )
        parser.add_argument(
            '--analysis', default=None, type=str,
            help=(
                'Top-level directory with the MaNGA DAP output. This is used to construct the '
                'output directory structure to mimic the SAS. If not defined and the direct root '
                'for the output is also not defined (see --root), this is set by the '
                'environmental variable MANGA_SPECTRO_ANALYSIS.'
            )
        ) 
        parser.add_argument(
            '--root', default=None, type=str,
            help='Output path for *all* files.  This overrides the default path construction.'
        )
        parser.add_argument(
            '-o', '--overwrite', default=False, action='store_true',
            help='Overwrite any existing files.'
        )

        return parser

    @classmethod
    def main(cls, args):
        manga.download_catalogs(
            dr=args.dr, oroot=args.root, redux_path=args.redux, analysis_path=args.analysis,
            overwrite=args.overwrite
        )
        return 0


