Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 12 items.

=== ITEM id=15c5548a0e18a734 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province.
  Fact 0.2: Liaoning province is in the People's Republic of China.
Sentence 1: Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.
  Fact 1.0: Kaiyuan is a county-level city.
  Fact 1.1: Kaiyuan is in the northeast of Liaoning.
  Fact 1.2: Liaoning is in the People's Republic of China.
  Fact 1.3: Kaiyuan borders Jilin for a small section to the north.

Retrieved items:
<item 1>
Kaiyuan West Railway Station
Kaiyuan West Railway Station is a railway station of Hada Passenger Railway and located in Liaoning, China
</item 1>

<item 2>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 2>

<item 3>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 3>

<item 4>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 4>

<item 5>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 5>

<item 6>
Entries describe Chinese and Korean places, people, and organizations: Pungdo is an inhabited Yellow Sea island in Ansan, South Korea. Liaoning locations
include Pulandian (2,770 km²; 741,230 people), Longkou (620,000), Kaiyuan, Qinghe District, Shuangta, Taiyang, Xingcheng, and Kaiyuan West station. Politicians
Du Zheheng (1910–75), Liaoning’s first governor
</item 6>

<item 7>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 7>

<item 8>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 8>

<item 9>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 9>

<item 10>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 10>
=== END ITEM id=15c5548a0e18a734 ===

=== ITEM id=5ed70ea2d7bd62c1 ===
Gold sentences and facts:
Sentence 0:  Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio Systems, Inc.
  Fact 0.0: The magazine was launched in 2006.
  Fact 0.1: The magazine is backed by Canadian industrialist David Harrison Gilmour.
  Fact 0.2: David Harrison Gilmour is the founder of FIJI Water LLC.
  Fact 0.3: The magazine is bimonthly.
  Fact 0.4: The magazine is distributed by Zinio Systems, Inc.
Sentence 1:  The first issue appeared in January-February 2007.
  Fact 1.0: The first issue of the magazine appeared in January-February 2007.
Sentence 2: La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine published from 1806 to 1837, founded by John Bell (1745–1831).
  Fact 2.0: La Belle Assemblée is also known in full as 'La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies'.
  Fact 2.1: La Belle Assemblée was a British women's magazine.
  Fact 2.2: La Belle Assemblée was published from 1806 to 1837.
  Fact 2.3: La Belle Assemblée was founded by John Bell.
  Fact 2.4: John Bell lived from 1745 to 1831.

Retrieved items:
<item 1>
Bogle-L'Ouverture Publications
Bogle-L’Ouverture Publications (BLP) is a radical London-based publishing company founded in 1968 by Guyanese activists Jessica Huntley (23 February 1927 – 13
October 2013) and Eric Huntley (born 25 September 1929).

 Named in honour of two outstanding liberation fighters in Caribbean history, Toussaint L'Ouverture and Paul Bogle, the company began operating during a period
in the UK when "books by Black authors or written with a sympathetic view of Black people’s history and culture were rare in mainstream bookshops in the UK."

 Alongside New Beacon Books (founded in 1966) and Allison & Busby (founded in 1967), BLP was one of the first black-owned independent publishing companies in
the UK.

 BLP has been described as "a small, unorthodox, self-financing venture that brought a radical perspective to non-fiction, fiction, poetry and children's
books."
</item 1>

<item 2>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 2>

<item 3>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 3>

<item 4>
Annales de chimie et de physique
Annales de chimie et de physique (French for "Annals of Chemistry and of Physics") is a scientific journal that was founded in Paris, France, in 1789 under the
title "Annales de chimie".

 One of the early editors was the French chemist Antoine Lavoisier.

 In 1815, it became the "Annales de chimie et de physique", and was published under that name for the next 100 years.
</item 4>

<item 5>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 5>

<item 6>
Louis Coues Page
Louis Coues Page (1869–1956) was a publisher in Boston, Massachusetts.

 Born in Zurich to American parents, he attended Harvard College and worked for Boston publishers Estes & Lauriat, 1891–1892.

 In 1896 he bought the Joseph Knight Company and renamed it L.C. Page & Company; around 1914 it became The Page Company.

 It issued works of "art, travel, music, belles lettres" and fiction for adults and children.

 It operated from offices on Beacon Street in Beacon Hill.

 Authors published by the firm included Bliss Carman, Julia Caroline Dorr, Lucy Maud Montgomery, and Eleanor H. Porter.

 In 1914 the Page Company acquired Dana Estes & Co.
</item 6>

<item 7>
La Belle Assemblée
La Belle Assemblée (in full La Belle Assemblée or, Bell's Court and Fashionable Magazine Addressed Particularly to the Ladies) was a British women's magazine
published from 1806 to 1837, founded by John Bell (1745–1831).
</item 7>

<item 8>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 8>

<item 9>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 9>

<item 10>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 10>
=== END ITEM id=5ed70ea2d7bd62c1 ===

=== ITEM id=38f91e3a516caa4a ===
Gold sentences and facts:
Sentence 0: The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil War.
  Fact 0.0: The Seven Days Battles were a series of six major battles.
  Fact 0.1: The Seven Days Battles were fought over the seven days from June 25 to July 1, 1862.
  Fact 0.2: The Seven Days Battles were fought near Richmond, Virginia.
  Fact 0.3: The Seven Days Battles took place during the American Civil War.
Sentence 1: The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.
  Fact 1.0: The Battle of Manila took place from February 3, 1945 to March 3, 1945.
  Fact 1.1: The Battle of Manila was a major battle of the Philippine campaign of 1944-45.
  Fact 1.2: The Battle of Manila took place during the Second World War.

Retrieved items:
<item 1>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 1>

<item 2>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 2>

<item 3>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 3>

<item 4>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 4>

<item 5>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 5>

<item 6>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 6>

<item 7>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 7>

<item 8>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 8>

<item 9>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 9>

<item 10>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 10>
=== END ITEM id=38f91e3a516caa4a ===

=== ITEM id=3e8d8ed6c839a0e4 ===
Gold sentences and facts:
Sentence 0: Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.
  Fact 0.0: Colocasia is a genus of flowering plants.
  Fact 0.1: Colocasia is in the family Araceae.
  Fact 0.2: Colocasia is native to southeastern Asia.
  Fact 0.3: Colocasia is native to the Indian subcontinent.
Sentence 1: The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
  Fact 1.0: The genus Coronilla contains about 20 species.
  Fact 1.1: The Coronilla species are flowering plants.
  Fact 1.2: The Coronilla species are native to Europe.
  Fact 1.3: The Coronilla species are native to North Africa.

Retrieved items:
<item 1>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 1>

<item 2>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 2>

<item 3>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 3>

<item 4>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 4>

<item 5>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 5>

<item 6>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 6>

<item 7>
Colocasia
Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.

 Some species are widely cultivated and naturalized in other tropical and subtropical regions.

 Common names include tarul, karkala ko ganu, elephant-ear, taro, cocoyam, dasheen, chembu, champadhumpa, shavige gadde, and eddoe.

 Elephant-ear and cocoyam are also used for some other large-leaved genera in the Araceae, notably "Xanthosoma" and "Caladium".

 The generic name is derived from the ancient Greek word "kolokasion", which in the Greek botanist Dioscorides (1st century AD) meant the edible roots of both
"Colocasia esculenta" and "Nelumbo nucifera".

 It is thought that the edible roots of "Colocasia esculenta" have been cultivated in Asia for more than ten thousand years.

 The species "Colocasia esculenta" is an invasive species in wetlands along the American Gulf coast, where it threatens to displace native wetland plants.
</item 7>

<item 8>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 8>

<item 9>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 9>

<item 10>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 10>
=== END ITEM id=3e8d8ed6c839a0e4 ===

=== ITEM id=624f6be55b1a4783 ===
Gold sentences and facts:
Sentence 0: Park Sung-woong (born January 9, 1973) is a South Korean actor.
  Fact 0.0: Park Sung-woong was born on January 9, 1973.
  Fact 0.1: Park Sung-woong is a South Korean actor.
Sentence 1:  Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a serial killer in "The Deal" (2015).
  Fact 1.0: Park Sung-woong made his acting debut in 'No. 3' in 1997.
  Fact 1.1: Park Sung-woong has starred in several movies and television series.
  Fact 1.2: Park Sung-woong notably starred as a gangster in 'New World' (2013).
  Fact 1.3: Park Sung-woong notably starred as a serial killer in 'The Deal' (2015).
Sentence 2: The Deal (; lit.
  Fact 2.0: The Deal's literal translation is given in the next sentence, after the 'lit.' marker.
Sentence 3:  "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
  Fact 3.0: The literal translation of The Deal is 'Murder Request'.
  Fact 3.1: The Deal is a 2015 South Korean crime thriller film.
  Fact 3.2: The Deal was directed by Son Yong-ho.

Retrieved items:
<item 1>
Office (2015 South Korean film)
Office (오피스 ) is 2015 South Korean slasher-thriller film directed by Hong Won-chan and starring Go Ah-sung and Park Sung-woong.

 It is about a detective trying to figure out why a mild-mannered man has killed his family and is targeting his co-workers.

 The film premiered at the Midnight Screenings section of the 2015 Cannes Film Festival.
</item 1>

<item 2>
Fatal Intuition
Fatal Intuition (; lit.

 "It's Him") is a 2015 South Korean crime thriller film written and directed by Yun Jun-hyeong.
</item 2>

<item 3>
A Company Man
A Company Man () is a 2012 South Korean action thriller film starring So Ji-sub, Lee Mi-yeon, Kwak Do-won and Kim Dong-jun.

 It is about a hitman who find himself targeted by his ex-employers after he falls in love with a single mother and quits his job.
</item 3>

<item 4>
A Violent Prosecutor
A Violent Prosecutor is a 2016 South Korean crime film directed by Lee Il-hyung, produced by Guk Su Ran and starring Hwang Jung-min, Kang Dong-won, Lee Sung-min
and Park Sung-woong.

 It was released in South Korea on February 3, 2016 by Showbox.
</item 4>

<item 5>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 5>

<item 6>
Deep Trap
Deep Trap () (previously known as Exchange) is a 2015 South Korean crime thriller film starring Ma Dong-seok, Jo Han-sun and Kim Min-kyung, and directed by Kwon
Hyung-jin.

 Based on an actual SNS crime, it depicts the terror experienced by a married couple while on a trip to an isolated island.

 It won Best Film Award in the Orient Express section at the Fantasporto in 2016.
</item 6>

<item 7>
The Boys Who Cried Wolf
The Boys Who Cried Wolf (), also known as The Shepherd, is a 2015 South Korean crime thriller drama film.

 Written and directed by Kim Jin-hwang in his first feature-length for his Korean Academy of Film Arts (KAFA)'s final year undergraduate film project, it
depicts how a former stage actor becomes involved in a murder case.
</item 7>

<item 8>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 8>

<item 9>
The Swindlers (2017 film)
The Swindlers is an upcoming South Korean crime drama film directed by Jang Chang-won.

 The film stars Hyun Bin, Yoo Ji-tae, Bae Seong-woo, Park Sung-woong, Nana and Ahn Se-ha.
</item 9>

<item 10>
Park Sung-woong
Park Sung-woong (born January 9, 1973) is a South Korean actor.

 Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a
serial killer in "The Deal" (2015).
</item 10>
=== END ITEM id=624f6be55b1a4783 ===

=== ITEM id=b0c399f702d545ef ===
Gold sentences and facts:
Sentence 0: The City Heiress is a play by Aphra Behn produced in 1682.
  Fact 0.0: The City Heiress is a play by Aphra Behn.
  Fact 0.1: The City Heiress was produced in 1682.
Sentence 1: Aphra Behn ( ; 14 December 1640?
  Fact 1.0: Aphra Behn's birth date is listed as 14 December 1640, marked with a question mark as uncertain.
Sentence 2:  (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.
  Fact 2.0: The 14 December 1640 date for Aphra Behn is a baptismal date.
  Fact 2.1: Aphra Behn died on 16 April 1689.
  Fact 2.2: Aphra Behn was a British playwright.
  Fact 2.3: Aphra Behn was a poet.
  Fact 2.4: Aphra Behn was a translator.
  Fact 2.5: Aphra Behn was a fiction writer.
  Fact 2.6: Aphra Behn was from the Restoration era.

Retrieved items:
<item 1>
Jules Wright
Jules Wright (25 February 1948 – 21 June 2015) was an Australian-born theatre director, a co-founder in 1984 of the Women’s Playhouse Trust (WPT), the first
resident woman director at the Royal Court Theatre, being only the second woman to direct on its main stage, and founder of the Wapping Project arts space.
</item 1>

<item 2>
The Autumn Garden
The Autumn Garden is a 1951 play by Lillian Hellman.

 The play is set in September, 1949 in a summer home in a resort on the Gulf of Mexico, about 100 miles from New Orleans.

 The play is a study of the defeats, disappointments and diminished expectations of people reaching middle age.

 For inspiration, Hellman drew on her memories of her time in her aunts' boardinghouse.

 Dashiell Hammett, who had been Hellman's lover for 20 years, helped her write the play and received 15 percent of the royalties.

 Of all Hellman's plays it was her favorite.
</item 2>

<item 3>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 3>

<item 4>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 4>

<item 5>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 5>

<item 6>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 6>

<item 7>
Graph query row:
person: Aphra Behn
death_date: 16 April 1689
relationship: CREATED_BY
creative_work: The City Heiress
</item 7>

<item 8>
Lillian Hellman: An Imperious Life
Lillian Hellman: An Imperious Life is a 2014 book by Dorothy Gallagher.

 It is a critical biography of the American playwright and writer Lillian Hellman.
</item 8>

<item 9>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 9>

<item 10>
The City Heiress
The City Heiress is a play by Aphra Behn produced in 1682.

 The play conforms to the general rules of Restoration comedy, but it also keeps Behn's own highly Royalist political point of view.
</item 10>
=== END ITEM id=b0c399f702d545ef ===

=== ITEM id=b93ede62d91d42e6 ===
Gold sentences and facts:
Sentence 0: Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.
  Fact 0.0: Dangal's English title is "Wrestling competition".
  Fact 0.1: Dangal is a 2016 film.
  Fact 0.2: Dangal is an Indian film.
  Fact 0.3: Dangal is a Hindi-language film.
  Fact 0.4: Dangal is a biographical sports drama film.
  Fact 0.5: Dangal is loosely based on the story of Mahavir Singh Phogat.
  Fact 0.6: Mahavir Singh Phogat is an amateur wrestler.
  Fact 0.7: Mahavir Singh Phogat trains his daughters Geeta Phogat and Babita Kumari.
  Fact 0.8: Mahavir Singh Phogat trains his daughters to be world-class wrestlers.
Sentence 1: The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
  Fact 1.0: The Man from Snowy River II is a 1988 film.
  Fact 1.1: The Man from Snowy River II is an Australian drama film.
  Fact 1.2: The Man from Snowy River II is the sequel to the 1982 film "The Man from Snowy River".

Retrieved items:
<item 1>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 1>

<item 2>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 2>

<item 3>
The Man from Snowy River and Other Verses
The Man from Snowy River and Other Verses (1895) is the first collection of poems by Australian poet Banjo Paterson.

 It was released in hardback by Angus and Robertson in 1895, and features the poet's widely anthologised poems "The Man from Snowy River", "Clancy of the
Overflow", "Saltbush Bill" and "The Man from Ironbark".

 It also contains the poet's first two poems that featured in The Bulletin Debate, a famous dispute in "The Bulletin" magazine from 1892-93 between Paterson and
Henry Lawson.
</item 3>

<item 4>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 4>

<item 5>
The Man from Snowy River II
The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
</item 5>

<item 6>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 6>

<item 7>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 7>

<item 8>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 8>

<item 9>
The Man from Snowy River (soundtrack)
"The Man from Snowy River is the original motion picture soundtrack from the 1982 film "The Man from Snowy River".
</item 9>

<item 10>
The Man from Snowy River (poem)
"The Man from Snowy River" is a poem by Australian bush poet Banjo Paterson.

 It was first published in "The Bulletin", an Australian news magazine, on 26 April 1890, and was published by Angus & Robertson in October 1895, with other
poems by Paterson, in "The Man from Snowy River and Other Verses".
</item 10>
=== END ITEM id=b93ede62d91d42e6 ===

=== ITEM id=cc0700cfd480904a ===
Gold sentences and facts:
Sentence 0: The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
  Fact 0.0: The International Grenfell Association (IGA) is an organization.
  Fact 0.1: The International Grenfell Association was founded by Sir Wilfred Grenfell.
  Fact 0.2: The International Grenfell Association provides health care.
  Fact 0.3: The International Grenfell Association provides education.
  Fact 0.4: The International Grenfell Association provides religious services.
  Fact 0.5: The International Grenfell Association provides rehabilitation.
  Fact 0.6: The International Grenfell Association provides other social activities.
  Fact 0.7: The International Grenfell Association serves fisherman and coastal communities in northern Newfoundland.
  Fact 0.8: The International Grenfell Association serves fisherman and coastal communities on the coast of Labrador.
Sentence 1: Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
  Fact 1.0: Sir Wilfred Thomason Grenfell held the title KCMG.
  Fact 1.1: Sir Wilfred Thomason Grenfell was born on 28 February 1865.
  Fact 1.2: Sir Wilfred Thomason Grenfell died on 9 October 1940.
  Fact 1.3: Sir Wilfred Thomason Grenfell was a medical missionary to Newfoundland.
  Fact 1.4: Sir Wilfred Thomason Grenfell was a medical missionary to Labrador.

Retrieved items:
<item 1>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 1>

<item 2>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 2>

<item 3>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 3>

<item 4>
John G. Wilson
John Guy Wilson (1842–1892) was a Manchester, UK patent agent and the founder of Wilson Gunn, one of the oldest firms of Patent and Trade Mark Attorneys in the
UK.

 John G. Wilson built up the business until his death in the 1890s.

 As a side line he acted as agent to the Alliance Insurance Company.

 There is no record that Mr. Wilson ever became a member of the Chartered Institute of Patent Agents.
</item 4>

<item 5>
Wilfred Grenfell
Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
</item 5>

<item 6>
Angelino Apelar
Angelino "Andy" Baldemor Apelar (June 16, 1927 - January 23, 2006) was an Evangelical Christian leader in the Filipino American community, and a founder of the
Association of Filipino Churches (AFC) of the Christian and Missionary Alliance.

 He had served as a missionary/pastor and radio broadcaster in the Philippines before moving to the United States to start the AFC.

 He served as president for several terms.

 He graduated from the Far Eastern Bible Institute and Seminary, where he also met his future wife, Purita Palomar, a member of one of Capiz province's
influential families.

 They have two children: Faith and Jemuel Apelar.
</item 6>

<item 7>
Paul Schulte
Paul Schulte OMI (1896 – 1975), was a German priest and missionary, known as the "Flying Priest", who founded MIVA ("Missionary International Vehicular
Association") to provide automobiles, boats and airplanes for the service of missions throughout the world.
</item 7>

<item 8>
Andrew Bryan (Baptist)
Andrew Bryan (1737–1812) founded First Bryan Baptist Church, affectionately called the Mother Church of Black Baptists, and First African Baptist Church of
Savannah in Savannah, Georgia, the first black Baptist churches to be established in America.

 Bryan was the former slave of Jonathan Bryan.
</item 8>

<item 9>
International Grenfell Association
The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and
rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
</item 9>

<item 10>
Godwin Birchenough
The Very Reverend Godwin Birchenough (27 October 1880, Macclesfield, Cheshire – 3 March, 1953) was the only son of Walter Edwin Birchenough and was the grandson
of John Birchenough, a prominent Macclesfield silk manufacturer.

 Godwin Birchenough, who was also a nephew of Sir Henry Birchenough, the President of the British South Africa Company, was educated at Rugby and Oriel College,
Oxford.

 Birchenough was ordained in 1905 and was Vicar of Moor Allerton between 1913 and 1921.

 He became an honorary Canon of Chelmsford Cathedral in 1933 and in 1941 became Dean of Ripon Cathedral, becoming Dean Emeritus in 1951.

 An eminent author, he was also vice chairman of the Additional Curates Society between 1934 and 1944.

 Godwin Birchenough married Edith, daughter of Ernest Keay in 1912, he died on 3 March 1953.
</item 10>
=== END ITEM id=cc0700cfd480904a ===

=== ITEM id=38262480c1748fb0 ===
Gold sentences and facts:
Sentence 0:  The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
  Fact 0.0: The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part.
  Fact 0.1: Sammy Hagar played the rhythm guitar part on a Gibson acoustic.
Sentence 1: OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist Sammy Hagar.
  Fact 1.0: OU812 is pronounced 'Oh You Ate One Too'.
  Fact 1.1: OU812 is the eighth studio album by American hard rock band Van Halen.
  Fact 1.2: OU812 was released in 1988.
  Fact 1.3: OU812 is the second Van Halen album to feature vocalist Sammy Hagar.

Retrieved items:
<item 1>
Hoarse &amp; Roaring
Hoarse & Roaring is the debut full-length album by neofolk band Parlor Hawk, released in 2010 on Northplatte Records.

 The album was produced by Joshua James and featured Neon Trees bassist Branden Campbell on each track (except "Saddest Song") as well as Fictionist member
Stuart Maxfield on the tracks "14 Years" and "Home".

 It was mixed by Todd Burke (Ben Harper, Jack Johnson) and mastered by Reuben Cohen (Edward Sharpe & The Magnetic Zeros, Bruno Mars).
</item 1>

<item 2>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 2>

<item 3>
Keisuke Kuwata (album)
Keisuke Kuwata is the eponymous studio album recorded by Keisuke Kuwata, a frontman of the Japanese rock band Southern All Stars.

 It was released by Taishita label under the Victor Entertainment in July 1988, shortly after the 10th anniversary of the band.

 Aside from the 1982 live recording album "Kamon Yuzo and Victor Wheels Live" released under the pseudonym Yūzō Kamon, it was first Kuwata's solo effort.
</item 3>

<item 4>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 4>

<item 5>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 5>

<item 6>
OU812 Tour
The OU812 Tour was a concert tour by hard rock band Van Halen.

 It was the second tour to include dates in Japan, and was the second with Sammy Hagar as vocalist (who had settled into the role more by this point).

 It followed the Monsters of Rock Tour 1988, which had formed the first part of the promotion for the "OU812" album.
</item 6>

<item 7>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 7>

<item 8>
Finish What Ya Started
"Finish What Ya Started" is a song by Van Halen taken from their 1988 album "OU812".

 Despite the album being seemingly complete, Eddie Van Halen came up with the riff at 2 in the morning and went down to his then-neighbor Sammy Hagar to show
it.

 Hagar let Eddie in, and the two played guitars in his balcony until they had a completed song.

 Once Eddie left, Hagar decided to write the lyrics despite being late at night.

 The theme wound up being unfulfilled sex, summed up by Hagar as "blue balls.

 In the song, Eddie Van Halen recorded his guitar part on a Fender Stratocaster plugged direct into the studio mixing console.

 The song is one of only two Van Halen tracks featuring Sammy Hagar playing a rhythm guitar part, which he played on a Gibson acoustic.
</item 8>

<item 9>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 9>

<item 10>
OU812
OU812 (pronounced "Oh You Ate One Too") is the eighth studio album by American hard rock band Van Halen, released in 1988, and the second to feature vocalist
Sammy Hagar.

 Van Halen started work on the album in September 1987 and completed it in April 1988, just one month before its release.
</item 10>
=== END ITEM id=38262480c1748fb0 ===

=== ITEM id=2533d8eeec0a02df ===
Gold sentences and facts:
Sentence 0: Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".
  Fact 0.0: Stephanocereus is a genus of cactus.
  Fact 0.1: Stephanocereus is from Brazil.
  Fact 0.2: Stephanocereus is related to "Arrojadoa".
Sentence 1:  They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).
  Fact 1.0: The plants in this sentence are mostly liana-forming.
  Fact 1.1: The plants in this sentence are found in the Old World tropics.
  Fact 1.2: The plants in this sentence range from South China, Indonesia, Malaysia and the Philippines.
  Fact 1.3: The plants in this sentence range westward to Madagascar, with two species there.
  Fact 1.4: The plants in this sentence range westward to the Seychelles, with one species there.
  Fact 1.5: The plants in this sentence range southward to Australia, with three species there.
  Fact 1.6: The plants in this sentence range southward to New Caledonia, with one species there.
  Fact 1.7: The plants in this sentence range northward to India, with one species there.
  Fact 1.8: The plants in this sentence range northward to Sri Lanka, with one species there.

Retrieved items:
<item 1>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 1>

<item 2>
Robert Cantley
Robert Cantley is a conservationist and Managing Director of Borneo Exotics, a Sri Lankan-based plant nursery specialising in tissue-cultured and seed-grown
"Nepenthes" species and hybrids.

 Cantley has contributed to a number of papers on "Nepenthes".

 The species "Nepenthes robcantleyi", the natural hybrid "Nepenthes × cantleyi" and the cultivar "Nepenthes" 'Cantley's Red' are named after him.
</item 2>

<item 3>
Nepenthes × kinabaluensis
Nepenthes × kinabaluensis , or the Kinabalu Pitcher-Plant, is the natural hybrid between "N. rajah" and "N. villosa".

 It was first collected near Kambarangoh on Mount Kinabalu, Borneo by Lilian Gibbs in 1910 and later mentioned by John Muirhead Macfarlane as ""Nepenthes" sp."

 in 1914.

 Although Macfarlane did not formally name the plant, he noted that "[a]ll available morphological details suggest that this is a hybrid between "N. villosa"
and "N. rajah"".

 It was finally described in 1976 by Shigeo Kurata as "N. × kinabaluensis".

 The name was first published in "Nepenthes of Mount Kinabalu", but was a "nomen nudum" at the time as it lacked an adequate description and information on the
type specimen.

 The name was subsequently published validly by Kurata in 1984.
</item 3>

<item 4>
Graph query row:
entity: Nepenthes
relationship: LOCATED_IN
related: Old World tropics
</item 4>

<item 5>
Nepenthes boschiana
Nepenthes boschiana ( ; after Johannes van den Bosch), or Bosch's Pitcher-Plant, is a tropical pitcher plant endemic to Borneo.

 It is most closely allied to "N. faizaliana".

 "Nepenthes borneensis" is considered a synonym of this species.

 "Nepenthes boschiana" has no known natural hybrids.

 No valid forms or varieties have been described.

 "Nepenthes boschiana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. chaniana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 5>

<item 6>
Nepenthes veitchii
Nepenthes veitchii ( or ; after James Veitch, nurseryman of the Veitch Nurseries), or Veitch's Pitcher-Plant, is a "Nepenthes" species from the island of
Borneo.

 The plant is widespread in north-western Borneo and can also be found in parts of Kalimantan.

 "Nepenthes veitchii" usually grows as an epiphyte, though the form from Bario seems to be strictly terrestrial and has not been observed to climb trees.
</item 6>

<item 7>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 7>

<item 8>
Nepenthes chaniana
Nepenthes chaniana ( ; after Datuk Chan Chew Lun, Managing Director of Natural History Publications (Borneo)) is a tropical pitcher plant species belonging to
the genus "Nepenthes".

 It is characterised by a dense indumentum of long, white hairs.

 Pitchers are cylindrical and mostly white to yellow in colouration.

 "Nepenthes chaniana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. boschiana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 8>

<item 9>
Nepenthes tenax
Nepenthes tenax ( ; from Latin: "tenax" "tenacious") is a lowland species of tropical pitcher plant native to northern Queensland, Australia.

 It is the third "Nepenthes" species recorded from the continent and its second endemic species.

 "Nepenthes tenax" is closely related to the two other Australian "Nepenthes" species: "N. mirabilis" and "N. rowaniae".
</item 9>

<item 10>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 10>
=== END ITEM id=2533d8eeec0a02df ===

=== ITEM id=3793aa73c0ad762d ===
Gold sentences and facts:
Sentence 0: "Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".
  Fact 0.0: 'Into Dust' is a song by the American duo Mazzy Star.
  Fact 0.1: 'Into Dust' appears as the ninth track on the second album of Mazzy Star.
  Fact 0.2: The second album of Mazzy Star is 'So Tonight That I Might See'.
Sentence 1: Mazzy Star is an American alternative rock band formed in Santa Monica, California, in 1989 from the group Opal, a collaboration of guitarist David Roback and bassist Kendra Smith.
  Fact 1.0: Mazzy Star is an American alternative rock band.
  Fact 1.1: Mazzy Star was formed in Santa Monica, California, in 1989.
  Fact 1.2: Mazzy Star was formed from the group Opal.
  Fact 1.3: Opal was a collaboration of guitarist David Roback and bassist Kendra Smith.

Retrieved items:
<item 1>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 1>

<item 2>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 2>

<item 3>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 3>

<item 4>
Assembly of Dust
Assembly of Dust (also known as AOD) is an American rock band formed in 2002 by former Strangefolk frontman Reid Genauer.

 After the breakup of Strangefolk, Genauer decided to record a solo album, and he recruited some friends from Strangefolk's days on the road to help out.

 After titling the 2003 solo release “Assembly of Dust,” Genauer decided to use the name for his new group.

 AOD originally featured Genauer on lead vocals and guitar, Nate Wilson on the keyboard, Adam Terrell on lead guitar, John Leccese on bass, and Andy Herrick on
drums.

 In recent years the band has experienced some transformations, and the current status of the band now features Reid Genauer on lead vocals and guitar, Adam
Terrell on lead guitar, John Leccese on bass, Jason Crosby on the keys and violin, and Dave Diamond on the drums.

 Despite the changes in the band's lineup, AOD has managed to maintain the same high energy and tight musical finesse for which they are known.
</item 4>

<item 5>
Berlin (band)
Berlin is an American new wave band.

 The group was formed in Orange County in 1979 by John Crawford (bass guitar).

 Band members included Crawford, Terri Nunn (vocals), David Diamond (keyboards), Ric Olsen (guitar), Matt Reid (keyboards) and Rod Learned (drums).

 The band gained mainstream-commercial success in the early 1980s with singles including "The Metro", "Sex (I'm A..

.)

", "No More Words" and then in the mid 1980s with the chart-topping single "Take My Breath Away" from the 1986 film "Top Gun".
</item 5>

<item 6>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 6>

<item 7>
Into Dust
"Into Dust" is a song by American duo Mazzy Star that appears as the ninth track on their second album "So Tonight That I Might See".

 The song's arrangement is sparse and features only acoustic guitar and violin as accompaniment to the vocals.
</item 7>

<item 8>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 8>

<item 9>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 9>

<item 10>
Onyx (Cornish band)
The Onyx or Onyx is a psychedelic rock band formed in Wadebridge, Cornwall, UK in 1965.

 Out of the ashes of Rick & The Hayseeds the band came to be known as The Onyx Set, named after an Onyx ring owned by original band member Mike Black-Borow.

 After various changes in the line-up they shortened their name to The Onyx and the classic line-up was formed.

 The band members were: guitarist Alan Hodge, who had previously played with various local bands such as The Buccaneers and The Fabulous Jaguars, vocalist Tony
Priest, bassist Dick Bland, keyboard player Steve Cotton and drummer Roger Dell.
</item 10>
=== END ITEM id=3793aa73c0ad762d ===

=== ITEM id=be8d930252b16ea9 ===
Gold sentences and facts:
Sentence 0: Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
  Fact 0.0: Black is a Bangladeshi rock band.
  Fact 0.1: Black was formed in Dhaka, Bangladesh.
  Fact 0.2: The Bengali name of Black is ব্ল্যাক.
Sentence 1: New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.
  Fact 1.0: New Found Glory is an American rock band.
  Fact 1.1: New Found Glory was formerly known as A New Found Glory.
  Fact 1.2: New Found Glory is from Coral Springs, Florida.
  Fact 1.3: New Found Glory was formed in 1997.

Retrieved items:
<item 1>
Nothing Gold Can Stay (album)
Nothing Gold Can Stay is the debut studio album by American rock band New Found Glory, released on October 19, 1999 through independent record label Eulogy
Recordings.

 At the time, the band was then named "A New Found Glory", but later dropped the indefinite article "A" due to some fans struggling to find their records in
stores.

 The original pressings of the album contained samplings from several films including "The Outsiders" (1983), "Weird Science" (1985), and "That Thing You Do!

" (1996), as well as Robert Frost's poem "Nothing Gold Can Stay", after which the album is named.
</item 1>

<item 2>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 2>

<item 3>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 3>

<item 4>
It's All About the Girls
It's All About the Girls is the debut EP by A New Found Glory (now New Found Glory) released on December 20, 1997 by Fiddler Records.

 It is the only release with their original drummer Joe Marino.

 A re-release, featuring revised cover artwork, was issued by Fiddler in June, 2000.
</item 4>

<item 5>
A Very New Found Glory Christmas
A Very New Found Glory Christmas is the third EP by American rock band New Found Glory.

 Self-produced by the band, it is a special edition Christmas-themed acoustic recording featuring two original songs and three covers.

 The EP was limited to 2,000 pressings and was released via cassette tape on December 1, 2012.

 1,000 white copies were sold on tour, and 1,000 red copies were sold in their online store.

 The band released "Nothing for Christmas" via Rdio and Spotify on December 20, after debuting songs off the EP during the tenth anniversary tour of their
"Sticks and Stones" album.

 The cover art is a parody of the 1990 blockbuster movie "Home Alone".
</item 5>

<item 6>
Radiosurgery (album)
Radiosurgery is the seventh studio album by American rock band New Found Glory.

 It was first released on September 30, 2011 in Australia, before its wider release on October 4 through independent label Epitaph Records.

 It is the band's final studio album to feature founding guitarist Steve Klein.

 To follow up predecessor "Not Without a Fight" (2009), the band began writing new material during their stint on the 2010 Honda Civic Tour.

 After self-producing a set of demos and contacting long-term record producer Neal Avron, the band went on to record the album in Avron's home recording studio
over a period of three months in 2011.

 The quintet set out to write an album that paid homage to classic punk rock records that first inspired them to form a band during the 1990s.

 Listening extensively to the likes of early Green Day and Ramones, New Found Glory strove to create a sound that could "bridge the gap" between old and new
generations of the genre.
</item 6>

<item 7>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
</item 7>

<item 8>
Hit or Miss (New Found Glory song)
"Hit or Miss (Waited Too Long)" is the first single by New Found Glory (formerly A New Found Glory), from their 1999 debut studio album, "Nothing Gold Can
Stay".
</item 8>

<item 9>
Black (Bangladeshi band)
Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
</item 9>

<item 10>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 10>
=== END ITEM id=be8d930252b16ea9 ===
