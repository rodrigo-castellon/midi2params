#!/bin/bash

# download the file ID

if ! [ -d "data" ]; then
    mkdir data
fi

echo "Use this script to download the tar'ed folder within \`data\` into our local copy of \`data\` and untar it. First argument is file ID (default: 1pe_2Xv99B_0_pSZqiUyIwK1p-yRWitGM; the indomain data tarball) and second argument is folder name (default: indomain)."
echo

if [ "$1" != "" ]; then
    ID=$1
else
    ID="1pe_2Xv99B_0_pSZqiUyIwK1p-yRWitGM"
fi
LINK='https://docs.google.com/uc?export=download&id='$ID

if [ "$2" != "" ]; then
    FNAME=$2
else
    FNAME="indomain"
fi

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $LINK -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$ID" -O "data/$FNAME.tgz" && rm -rf /tmp/cookies.txt

cd data && tar -xvf "$FNAME.tgz"
