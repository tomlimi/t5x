!/bin/bash


LANGUAGES=('de' 'el' 'eo' 'et' 'eu' 'fa' 'fi' 'fy' 'ga' 'gd' 'gl' 'gu' 'ha' 'haw' 'hi' 'hmn' 'ht' 'hu' 'hy' 'id' 'ig' 'is' 'iw' 'jv' 'ka' 'kk' 'km' 'kn' 'ku' 'ky' 'la' 'lb' 'lo' 'lt' 'lv' 'mg' 'mi' 'mk' 'ml' 'mn' 'mr' 'ms' 'my' 'ne' 'nl' 'no' 'ny' 'pa' 'ps' 'ru' 'sd' 'sk' 'sl' 'sm' 'sn' 'so' 'sq' 'st' 'su' 'sv' 'tg' 'th' 'tr' 'uk' 'ur' 'uz' 'vi' 'xh' 'yi' 'yo' 'zh' 'zu' 'en' 'es' 'pt' 'fr' 'it' 'ro' 'pl' 'mt' 'he' 'ja' 'ko' 'te' 'ta' 'si')

for lang in "${LANGUAGES[@]}"
do
    echo "Processing $lang"
    screen -md -S "${lang}" ./process_language_data.sh $lang
done
