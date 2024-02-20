#!/bin/bash

################################################################################
############################# GDrive Downloader ################################
################################################################################
# The following is a helper function to download tarballs from the google
# drive and then untar them. This will mainly be used to download features
# and datasets.

function gdluntar {
  # usage: gdluntar FILEID FILENAME FILEURL PTHMD5FILE RMAFTERDL
  MYFILEID=$1
  MYTARFILE=$2
  MYURL=$3
  PTHMD5FILE=$4
  RMAFTERDL=$5
  ISSMALL=$6
  
  [[ "aa"$RMAFTERDL == "aa" ]] && echo "args incomplete" && return 1
  [[ "aa"$ISSMALL == "aa" ]] && ISSMALL=0

  if md5sum -c ${PTHMD5FILE} 2> /dev/null; then
    :
  else
    echo "The files in $PTHMD5FILE are either missing or corrupted."
    echo "Attempting a fresh download..."
    echo "Downloading: $MYURL -> $(readlink -m $MYTARFILE)"
    sleep 0.5
    if [[ $ISSMALL == "1" ]]; then
      wget --no-check-certificate 'https://docs.google.com/uc?export=download&id='"${MYFILEID}" -O ${MYTARFILE} 
    else
      wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${MYFILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${MYFILEID}" -O ${MYTARFILE} && rm -rf /tmp/cookies.txt
    fi

    if [[ ! -f $MYTARFILE ]]; then
      echo "Downloading Error: could not download the file $MYTARFILE."
      echo "  Perhaps this script is being outdated or gdrive API got updated."
      echo "  Please follow the following steps:"
      echo "    1. Download the file from the following URL:"
      echo "       URL ==> $MYURL"
      echo "       and place it at the following path"
      echo "       Path ==> $(readlink -m $MYTARFILE)"
      echo "    2. Run the following script:"
      echo "       tar -xvf $(readlink -m $MYTARFILE)"
      echo "    3. Get rid of the downloaded tarballs to save space:"
      echo "       rm $(readlink -m $MYTARFILE)"
      echo "    4. [Optional]: check the md5sums of the extracted files:"
      echo "       md5sum $(readlink -m $PTHMD5FILE)"
      echo "*******************************************************************"
    elif [[ $MYTARFILE == *.tar || $MYTARFILE == *.tar.gz ]]; then
      echo "----------";
      echo "Untarring ${MYTARFILE}:"
      [[ $MYTARFILE == *.tar ]] && tar -xvf $MYTARFILE
      [[ $MYTARFILE == *.tar.gz ]] && tar -xzvf $MYTARFILE
      [[ $RMAFTERDL == 1 ]] && rm $MYTARFILE
      echo "----------";
      if md5sum -c ${PTHMD5FILE}; then
        echo "";
        echo " ==> Downloading and untarring ${MYTARFILE} was sucesseful!"
      else
        echo "";
        echo "==> After extracting the md5sum checks did not match. "
        echo "    Either the downloaded file was corrupted "
        echo "    or the untarring went wrong or sth fishy happend."
        echo "    (most probably, of the first two)."
      fi
      echo "*****************************************************************"
      echo "";
    fi
  fi
}

################################################################################
############################### PATH Functions #################################
################################################################################

# SYNOPSIS: field_prepend varName fieldVal [sep]
field_prepend() {
    local varName=$1 fieldVal=$2 IFS=${3:-':'} auxArr
    read -ra auxArr <<< "${!varName}"
    for i in "${!auxArr[@]}"; do
        [[ ${auxArr[i]} == "$fieldVal" ]] && unset auxArr[i]
    done
    auxArr=("$fieldVal" "${auxArr[@]}")
    printf -v "$varName" '%s' "${auxArr[*]}"
}

# SYNOPSIS: field_append varName fieldVal [sep]
field_append() {
    local varName=$1 fieldVal=$2 IFS=${3:-':'} auxArr
    read -ra auxArr <<< "${!varName}"
    for i in "${!auxArr[@]}"; do
        [[ ${auxArr[i]} == "$fieldVal" ]] && unset auxArr[i]
    done
    auxArr+=("$fieldVal")
    printf -v "$varName" '%s' "${auxArr[*]}"
}
