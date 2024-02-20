#!/bin/bash

# Usage: ./download.sh
# note:  This script downloads the extracted features used in the distribution
#        calibration experiments, including the cross-dataset experiments and 
#        other ones.

SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPTDIR

# importing the gdluntar helper function from the utils directory
source ../utils/bashfuncs.sh

FILEID="1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO"
FILENAME="features.tar"
GDRIVEURL="https://drive.google.com/file/d/1nf_WeD7fcEAu2BLD-FLfKRaAtcoseSoO/view?usp=sharing"
PTHMD5FILE="features.md5"
REMOVETARAFTERDL="1"
ISSMALL="0"
gdluntar ${FILEID} ${FILENAME} ${GDRIVEURL} ${PTHMD5FILE} ${REMOVETARAFTERDL} ${ISSMALL}
