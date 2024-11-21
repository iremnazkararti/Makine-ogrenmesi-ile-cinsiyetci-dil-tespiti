import pandas as pd

# Temizlenmiş veri setini yükleyin
df_temizlenmis = pd.read_excel(r'C:\Users\Naz\PycharmProjects\bitirme2\temizlenmis_veri_seti.xlsx')

# 'Alt Etiket' sütunun adını 'Cinsiyetçi' olarak değiştirin
df_temizlenmis.rename(columns={'Alt Etiket': 'Cinsiyetçi'}, inplace=True)

# Cinsiyetçi ve diğer etiketler için koşullar belirleyin
def classify(row):
    # Cinsiyetçi tweetleri '1' ile etiketle
    if 'kadın' in row['Tweet'].lower() or 'erkek' in row['Tweet'].lower() or 'cinsiyet' in row['Tweet'].lower():
        return 1
    # Diğer etnik temalı tweetlere '0' yaz
    elif 'ırk' in row['Tweet'].lower() or 'etnik' in row['Tweet'].lower() or 'ağırlıklı' in row['Tweet'].lower():
        return 0
    else:
        return 0  # Varsayılan olarak diğer etiketlere 0 yazabiliriz

# 'Cinsiyetçi' sütununu oluşturun ve bu sütunda yukarıdaki sınıflandırmayı uygulayın
df_temizlenmis['Cinsiyetçi'] = df_temizlenmis.apply(classify, axis=1)

# Düzenlenmiş veri setini kaydedin
output_path = r'C:\Users\Naz\PycharmProjects\bitirme2\temizlenmis_veri_seti_etiketli.xlsx'
df_temizlenmis.to_excel(output_path, index=False)

print("Alt etiket 'Cinsiyetçi' olarak değiştirildi ve etiketleme yapıldı.")
