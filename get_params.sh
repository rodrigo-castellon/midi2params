#!/bin/bash
rm -rf params
mkdir params
pushd params

# Download parameters
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15XZ8FkWWRQuDaZJJff-77590OC7ly16t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15XZ8FkWWRQuDaZJJff-77590OC7ly16t" -O Flute.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Fg_s8WOsp8XO36plE455g34r39bBTr3r' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Fg_s8WOsp8XO36plE455g34r39bBTr3r" -O Flute2.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oSYIFXhOizcGvPJBl0K7Gnw2qPkUgo-v' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oSYIFXhOizcGvPJBl0K7Gnw2qPkUgo-v" -O Tenor_Saxophone.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10JJqFYeYNzIkJPcFqyPC5ImNGZXoUJJj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10JJqFYeYNzIkJPcFqyPC5ImNGZXoUJJj" -O Trumpet.tar.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a1JlCsBfvw_qs-feL8KIJGtn_teA8xg2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a1JlCsBfvw_qs-feL8KIJGtn_teA8xg2" -O Violin.tar.gz && rm -rf /tmp/cookies.txt
sha256sum *

# Extract and organize
for TAG in Flute Flute2 Tenor_Saxophone Trumpet Violin
do
	tar xvfz ${TAG}.tar.gz
	mkdir -p ddsp_official/${TAG}
	mv pretrained/* ddsp_official/${TAG}/
	rm ${TAG}.tar.gz
done
rm -rf pretrained

popd
