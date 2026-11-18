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

=== ITEM id=09896b3c8ea71c8e ===
Gold sentences and facts:
Sentence 0:  Game included wild boar, and red, fallow and roe deer.
  Fact 0.0: Game included wild boar.
  Fact 0.1: Game included red deer.
  Fact 0.2: Game included fallow deer.
  Fact 0.3: Game included roe deer.
Sentence 1:  This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.
  Fact 1.0: The common species is native to western Eurasia.
  Fact 1.1: The common species has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar, Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the United States, the Falkland Islands, and Peru.

Retrieved items:
<item 1>
Red deer
The red deer ("Cervus elaphus") is one of the largest deer species.

 The red deer inhabits most of Europe, the Caucasus Mountains region, Asia Minor, Iran, parts of western Asia, and central Asia.

 It also inhabits the Atlas Mountains region between Morocco and Tunisia in northwestern Africa, being the only species of deer to inhabit Africa.

 Red deer have been introduced to other areas, including Australia, New Zealand, United States, Canada, Peru, Uruguay, Chile and Argentina.

 In many parts of the world, the meat (venison) from red deer is used as a food source.
</item 1>

<item 2>
Fallow deer
The fallow deer ("Dama dama") is a ruminant mammal belonging to the family Cervidae.

 This common species is native to western Eurasia, but has been introduced to Antigua & Barbuda, Argentina, South Africa, Fernando Pó, São Tomé, Madagascar,
Mauritius, Mayotte, Réunion, Seychelles, Comoro Islands, Morocco, Algeria, Tunisia, Cyprus, Israel, Cape Verde, Lebanon, Australia, New Zealand, Canada, the
United States, the Falkland Islands, and Peru.

 Some taxonomers include the rarer Persian fallow deer as a subspecies ("D. d. mesopotamica"), while others treat it as an entirely different species ("D.
mesopotamica").
</item 2>

<item 3>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 3>

<item 4>
Indian muntjac
The Indian muntjac ("Muntiacus muntjak"), also called red muntjac and barking deer, is a common muntjac deer species in South and Southeast Asia.

 It is listed as Least Concern on the IUCN Red List.
</item 4>

<item 5>
Forests of Mara and Mondrem
The Forests of Mara and Mondrem were adjacent medieval forests in Cheshire, England, which in the 11th century extended to over 60 sqmi , stretching from the
Mersey in the north almost to Nantwich in the south, and from the Gowy in the west to the Weaver in the east.

 Mara and Mondrem were a hunting forest of the Norman Earls of Chester, established soon after 1071 by the first earl, Hugh d'Avranches.

 They might earlier have been an Anglo-Saxon hunting forest.

 Game included wild boar, and red, fallow and roe deer.
</item 5>

<item 6>
Sri Lankan sambar deer
The Sri Lankan sambar deer ("Rusa unicolor unicolor") is a subspecies of sambar deer that lives in Sri Lanka.

 This subspecies is one of the largest sambar deer species with the largest antlers both in size and in body proportions.

 Large males weight up to 270–280 kg.

 Sri Lankan sambar live in lowland dry forests and mountain forests.

 Large herds of sambar deer roam the Horton Plains National Park, where it is the most common large mammal.
</item 6>

<item 7>
Visayan spotted deer
The Visayan spotted deer ("Rusa alfredi"), also known as the Philippine spotted deer, is a nocturnal and endangered species of deer located primarily in the
rainforests of the Visayan islands of Panay and Negros though it once roamed other islands such as Cebu, Guimaras, Leyte, Masbate, and Samar.

 It is one of three endemic deer species in the Philippines, although it was not recognized as a separate species until 1983.

 An estimated 2,500 mature individuals survived worldwide as of 1996, according to the IUCN, although it is uncertain of how many of them still survive in the
wild.

 The diet of the deer, which consists of a variety of different types of grasses, leaves, and buds within the forest, is the primary indicator of its habitat.

 Since 1991 the range of the species has severely decreased and is now almost co-extensive with that of the Visayan warty pig.
</item 7>

<item 8>
Marsh deer
The marsh deer ("Blastocerus dichotomus") is the largest deer species from South America reaching a length of 2 m and a shoulder height of 1.2 m .

 It is found in Argentina, Bolivia, Peru, Brazil, Uruguay and Paraguay.

 Formerly found in much of tropical and subtropical South America, it ranged east of the Andes, south from the Amazon rainforest, west of the Brazilian Atlantic
rainforest and north of the Argentinian Pampa.

 Today it is largely reduced to isolated populations at marsh and lagoon zones in the Paraná, Paraguay, Araguaia and Guapore river basins.

 Small populations also occur in the southern Amazon, including Peru where protected in Bahuaja-Sonene National Park.

 It is listed as a vulnerable species by the IUCN and on CITES Appendix I.
</item 8>

<item 9>
Moose
The moose (North America) or elk (Eurasia), "Alces alces", is the largest extant species in the deer family.

 Moose are distinguished by the broad, flat (or "palmate") antlers of the males; other members of the family have antlers with a dendritic ("twig-like")
configuration.

 Moose typically inhabit boreal forests and temperate broadleaf and mixed forests of the Northern Hemisphere in temperate to subarctic climates. Hunting and
other human activities have caused a reduction in the size of the moose's range over time.

 Moose have been reintroduced to some of their former habitats.

 Currently, most moose are found in Canada, Alaska, New England, Fennoscandia, Baltic States, and Russia.

 Their diet consists of both terrestrial and aquatic vegetation.

 The most common moose predators are the gray wolf along with bears and humans. Unlike most other deer species, moose are solitary animals and do not form
herds.

 Although generally slow-moving and sedentary, moose can become aggressive and move quickly if angered or startled.

 Their mating season in the autumn features energetic fights between males competing for a female.
</item 9>

<item 10>
White-bellied musk deer
The White-bellied musk deer or Himalayan musk deer ("Moschus leucogaster") is a musk deer species occurring in the Himalayas of Nepal, Bhutan, India, Pakistan
and China.

 It is listed as endangered by the IUCN because of overexploitation resulting in a probable serious population decline.
</item 10>
=== END ITEM id=09896b3c8ea71c8e ===

=== ITEM id=c4ac2becab0f2a30 ===
Gold sentences and facts:
Sentence 0: Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.
  Fact 0.0: Broughtonia is a genus of orchids.
  Fact 0.1: Broughtonia belongs to the family Orchidaceae.
  Fact 0.2: Broughtonia is native to the Bahamas.
  Fact 0.3: Broughtonia is native to the Greater Antilles.
Sentence 1: Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family, Lamiaceae.
  Fact 1.0: Perilla is a genus.
  Fact 1.1: Perilla consists of one major Asiatic crop species, P. frutescens.
  Fact 1.2: Perilla includes a few recognized wild species in nature.
  Fact 1.3: Perilla belongs to the mint family, Lamiaceae.

Retrieved items:
<item 1>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 1>

<item 2>
Shiso
Perilla frutescens var.

 "crispa", also called shiso ( , from Japanese シソ ) is a variety of species "Perilla frutescens" of the genus "Perilla", belonging to the mint family,
Lamiaceae.

 The plant occurs in red (purple-leaved) or green-leaved forms.

 Shiso is a perennial plant that may be cultivated as an annual in temperate climates, and occurs in both red- (or purple-) leaved and green forms.

 There are also frilly, ruffled-leaved forms called "chirimen-jiso" and forms that are red only on top, called "katamen-jiso".
</item 2>

<item 3>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 3>

<item 4>
Lamiaceae
or Labiatae is a family of flowering plants commonly known as the mint or deadnettle family.

 Many of the plants are aromatic in all parts and include widely used culinary herbs, such as basil, mint, rosemary, sage, savory, marjoram, oregano, hyssop,
thyme, lavender, and perilla.

 Some species are shrubs, trees (such as teak), or, rarely, vines.

 Many members of the family are widely cultivated, not only for their aromatic qualities but also their ease of cultivation, since they are readily propagated
by stem cuttings.

 Besides those grown for their edible leaves, some are grown for decorative foliage, such as "Coleus".

 Others are grown for seed, such as "Salvia hispanica" (chia), or for their edible tubers, such as "Plectranthus edulis", "Plectranthus esculentus",
"Plectranthus rotundifolius", and "Stachys affinis" (Chinese artichoke).
</item 4>

<item 5>
Graph query row:
entity: Perilla
relationship: PART_OF
related: Lamiaceae
</item 5>

<item 6>
Perilla
Perilla is a genus consisting of one major Asiatic crop species "P. frutescens" and a few recognized wild species in nature, belonging to the mint family,
Lamiaceae.

 It encompasses several distinct varieties of Asian herb, seed, and vegetable crop, including "P. frutescens" (deulkkae) and "P. frutescens" var.

 "crispa" (shiso).

 The genus name "Perilla" is also a frequently employed common name ("perilla"), applicable to all varieties.

 Perilla varieties are cross-fertile and intra-specific hybridization occurs naturally.

 Some varieties are considered invasive.
</item 6>

<item 7>
Broughtonia
Broughtonia is a genus of orchids (family Orchidaceae) native to the Bahamas and the Greater Antilles.

 The genus is abbreviated Bro in trade journals.
</item 7>

<item 8>
Perilla ketone
Perilla ketone is a natural terpenoid that consists of a furan ring with a six-carbon side chain containing a ketone functional group.

 It is a colorless oil that is sensitive to oxygen, becoming colored upon standing.

 The ketone was identified in 1943 by Sebe as the main component of the essential oil of Perilla frutescens.

 Perilla ketone is present in the leaves and seeds of purple mint ("Perilla frutescens"), which is toxic to some animals.

 When cattle and horses consume purple mint when grazing in fields in which it grows, the perilla ketone causes pulmonary edema leading to a condition sometimes
called perilla mint toxicosis.
</item 8>

<item 9>
Perilla frutescens
Perilla frutescens, commonly called perilla or Korean perilla, is a species of "Perilla" in the mint family Lamiaceae.

 It is an annual plant native to Southeast Asia and Indian highlands, and grown in the Korean peninsula, Southern China and India.

 Known as "deulkkae" (들깨 ) in Korean, the plant was introduced into Korea before the Unified Silla era, when it started to be widely cultivated.
</item 9>

<item 10>
Monardella lanceolata
Monardella lanceolata is a species of flowering plant in the mint family known by the common names mustang mint and mustang monardella.

 It is native to the mountains of California and Baja California, where it grows in chaparral, woodland, rocky slopes, and often disturbed habitat types.
</item 10>
=== END ITEM id=c4ac2becab0f2a30 ===

=== ITEM id=013b0047e8571030 ===
Gold sentences and facts:
Sentence 0:  She was born in Guangzhou, China.
  Fact 0.0: The woman referred to as "she" was born in Guangzhou, China.
Sentence 1:  These videos are an overview and recap of each year's viral videos, events, memes, and music.
  Fact 1.0: The videos being described are an overview of each year's viral videos, events, memes, and music.
  Fact 1.1: The videos being described are a recap of each year's viral videos, events, memes, and music.

Retrieved items:
<item 1>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 1>

<item 2>
Wayne Goss (make-up artist)
Wayne Alan Goss (born 4 March 1978) is an English makeup artist, YouTube personality and entrepreneur.

 Best known for his "Wayne Goss" channel on YouTube, he is the creator and owner of the self-eponymous makeup brush line launched in September 2013.

 As of November 2016, Goss' main YouTube channel had over 2.7 million subscribers and had had 344 million views.
</item 2>

<item 3>
Wengie
Wendy Ayche (born January 9 1986) known professionally as Wengie, is an Australian YouTube personality and vlogger.

 She was born in Guangzhou, China.

 She got the nickname "Wengie" at a dance class when two people gave her a nickname inspired by her Chinese name, Wén Jié.

 Since starting her channel on February 11 2013, her videos have received over 281 million views, and her channel has accumulated over 11 followers In the
January 2016 edition of "ElleGirl Japan", she was nominated as a channel to watch.

 In August 2016, she was ranked the 5th Fastest growing channel in the world.

 Ayche was featured in the annual YouTube Rewind in 2016 and was also singled out by Google as one of the top beauty creators in the Asia region.

 Ayche reached 5 million YouTube subscribers in January 2017, and her channel is currently in the 12th most subscribed How-To & Style Channel on YouTube.

 She is also currently a board member of the Internet Creators Guild, a non-profit focused on providing the protection, representation and guidance to online
creators.
</item 3>

<item 4>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 4>

<item 5>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 5>

<item 6>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 6>

<item 7>
Lilly Singh
Lilly Singh (born 26 September 1988) is a Canadian YouTube personality, vlogger, comedian, writer, and actress.

 She is better known by her YouTube username IISuperwomanII.

 Since beginning her channel in October 2010, her videos have received over 2 billion views, and her channel has accumulated over 12 million subscribers.

 In 2016, she was ranked 3rd on the "Forbes" list of the world's highest paid YouTube stars (behind Roman Atwood and PewDiePie), earning a reported $7.5 million
in 2016.

 Singh has featured in the annual YouTube Rewind every year since 2014.

 She ranked 1st on 2017 Forbes Top Influencers List in the entertainment category.
</item 7>

<item 8>
CVX Live
CVX Live (Creator Viewer Experience Live) is an event designed to bring online media creators and their fans together.

 It was held for the first time in Orem, Utah, on August 7 to 8, 2015 Originally created by the team behind the "Bored Shorts TV" YouTube channel, the
convention is Utah's first YouTube event, gathering thousands of online YouTube creators and fans together in one place.

 One of the creators, Brett Robers, described it as “...a very unique opportunity to bring YouTube to life.”
</item 8>

<item 9>
YouTube Rewind
YouTube Rewind is a video series produced and created by YouTube and Portal A Interactive.

 These videos are an overview and recap of each year's viral videos, events, memes, and music.

 Each year, the number of YouTube celebrities featured in the video, as well as the presentation of the series, have increased.

 The latest episode of "YouTube Rewind" was released on December 8, 2016.
</item 9>

<item 10>
Wendy Austin
Dr Wendy Elizabeth Austin Hewitt, Wendy Austin is a BBC journalist and broadcaster in BBC Northern Ireland, and former presenter of BBC Radio Ulster’s current
affairs programme "Talkback".
</item 10>
=== END ITEM id=013b0047e8571030 ===

=== ITEM id=29a9cb2959c73f90 ===
Gold sentences and facts:
Sentence 0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great Britain and is dedicated to Saint Chad of Mercia.
  Fact 0.0: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham.
  Fact 0.1: The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the province of the Catholic Church in Great Britain.
  Fact 0.2: The Metropolitan Cathedral Church and Basilica of Saint Chad is dedicated to Saint Chad of Mercia.
Sentence 1:  Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English Reformation and raised to cathedral status in 1852.
  Fact 1.0: St Chad's was built by Augustus Welby Pugin.
  Fact 1.1: St Chad's was substantially complete by 1841.
  Fact 1.2: St Chad's is one of the first four Catholic churches constructed after the English Reformation.
  Fact 1.3: St Chad's was raised to cathedral status in 1852.
Sentence 2: Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.
  Fact 2.0: Chelmsford Cathedral is in the city of Chelmsford, Essex, England.
  Fact 2.1: Chelmsford Cathedral is dedicated to St Mary the Virgin.
  Fact 2.2: Chelmsford Cathedral is dedicated to St Peter.
  Fact 2.3: Chelmsford Cathedral is dedicated to St Cedd.
Sentence 3:  It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
  Fact 3.0: Chelmsford Cathedral became a cathedral when the Anglican Diocese of Chelmsford was created in 1914.
  Fact 3.1: Chelmsford Cathedral is the seat of the Bishop of Chelmsford.

Retrieved items:
<item 1>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 1>

<item 2>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 2>

<item 3>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 3>

<item 4>
St Chad's College, Durham
St Chad's College is a recognised (independent) college of Durham University in England, founded in 1904 as an Anglican hall for the training of Church of
England clergy.

 The main part of the college is located on the Bailey, occupying nine historic buildings at the east end of Durham Cathedral.

 It neighbours Hatfield College to its north, while St John's College and St Cuthbert's Society are to its south.

 The college is named after St Chad of Mercia, a 7th-century bishop.
</item 4>

<item 5>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 5>

<item 6>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 6>

<item 7>
Bishop's House, Birmingham
The Bishop's House in Birmingham, England was designed by Augustus Pugin as the residence of Thomas Walsh, the first Roman Catholic Bishop of Birmingham.

 It was situated opposite St Chad's Cathedral, on the corner of Bath Street and Weaman Street in Birmingham City Centre.
</item 7>

<item 8>
St Chad's Cathedral, Birmingham
The Metropolitan Cathedral Church and Basilica of Saint Chad is the mother church of the Archdiocese of Birmingham and province of the Catholic Church in Great
Britain and is dedicated to Saint Chad of Mercia.

 Built by Augustus Welby Pugin and substantially complete by 1841, St Chad's is one of the first four Catholic churches that were constructed after the English
Reformation and raised to cathedral status in 1852.

 It is one of only four minor basilicas in England (the others being Downside Abbey, the National Shrine of Our Lady at Walsingham and Corpus Christi Priory,
this last now disused).

 St Chad's is a Grade II* listed building.

 The cathedral is located in a public greenspace near St Chad's Queensway, in central Birmingham.

 s of 2014 the Archbishop was Bernard Longley and the Dean Canon Gerry Breen.
</item 8>

<item 9>
Lichfield Cathedral
Lichfield Cathedral is situated in Lichfield, Staffordshire, England.

 It is the only medieval English cathedral with three spires.

 The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.

 The 99th and current Bishop of Lichfield is Michael Ipgrave who was appointed on 10 June 2016.
</item 9>

<item 10>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 10>
=== END ITEM id=29a9cb2959c73f90 ===

=== ITEM id=2a80b6891b9d5c2e ===
Gold sentences and facts:
Sentence 0: Rivermaya is a Filipino rock band.
  Fact 0.0: Rivermaya is a Filipino rock band.
Sentence 1: Kodaline ( ) are an Irish band.
  Fact 1.0: Kodaline is an Irish band.

Retrieved items:
<item 1>
Tanglewood Music and Arts Festival
Tanglewood Music and Arts Festival is an annual two-day music, comedy and arts event held within the grounds of Narrow Water Castle, County Down, Northern
Ireland.

 The festival, founded in 2011, has attracted significant attention both within the local area and around Northern Ireland with buses coming from as far as
Belfast to the event.

 For the 2011 and 2012 festivals Tanglewood was held on one day; however for 2013 it has expanded into a two-day camping festival.
</item 1>

<item 2>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 2>

<item 3>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 3>

<item 4>
Kodaline
Kodaline ( ) are an Irish band.

 Originally known as 21 Demands, in 2012 the band changed their name to Kodaline.
</item 4>

<item 5>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 5>

<item 6>
Ciaran Gribbin
Ciaran Gribbin (born 1976) is a Northern Irish singer-songwriter, guitarist and producer who hails from Castledawson, Northern Ireland.
</item 6>

<item 7>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 7>

<item 8>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 8>

<item 9>
Honest (Kodaline song)
"Honest" is a song by Dublin-based alternative rock quartet Kodaline.

 The song was released on 1 February 2015 as the lead single from the band's second studio album, "Coming Up for Air" (2015).

 "Honest" became the band's third top 10 single in their home country, Ireland, following "High Hopes" (2013) and "Love Like This" (2013).
</item 9>

<item 10>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 10>
=== END ITEM id=2a80b6891b9d5c2e ===

=== ITEM id=6a1ea3bc696d1811 ===
Gold sentences and facts:
Sentence 0: Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.
  Fact 0.0: Lowell J. Sherman was born on October 11, 1885.
  Fact 0.1: Lowell J. Sherman died on December 28, 1934.
  Fact 0.2: Lowell J. Sherman was an American actor.
  Fact 0.3: Lowell J. Sherman was an American film director.
Sentence 1: Jonathan Kaplan (born November 25, 1947) is an American film producer and director.
  Fact 1.0: Jonathan Kaplan was born on November 25, 1947.
  Fact 1.1: Jonathan Kaplan is an American film producer.
  Fact 1.2: Jonathan Kaplan is an American film director.

Retrieved items:
<item 1>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 1>

<item 2>
Lillian Hellman
Lillian Florence Hellman (June 20, 1905 – June 30, 1984) was an American dramatist and screenwriter known for her success as a playwright on Broadway, as well
as her left-wing sympathies and political activism.

 She was blacklisted after her appearance before the House Committee on Un-American Activities (HUAC) at the height of the anti-communist campaigns of 1947–52.

 Although she continued to work on Broadway in the 1950s, her blacklisting by the American film industry caused a drop in her income.

 Many praised Hellman for refusing to answer questions by HUAC, but others believed, despite her denial, that she had belonged to the Communist Party.
</item 2>

<item 3>
Lev Kuleshov (1899–1970), Soviet filmmaker and theorist, pioneered Soviet montage, the Kuleshov effect, and creative geography. Hermann Brunn (1862–1939),
German mathematician, contributed to convex geometry and knot theory. Mercury’s 165-km Chaikovskij crater honors composer Pyotr Tchaikovsky; a 1970 Soviet
biopic about him earned two Oscar nominations. Søren Kierkegaard (
</item 3>

<item 4>
Lawful Larceny
Lawful Larceny is a 1930 American melodramatic film, directed by Lowell Sherman from Jane Murfin's screenplay.

 The screenplay, a melodrama, was based on the play of the same name by Samuel Shipman, which originally was a comedy.

 It starred a staple of the early RKO stable, Bebe Daniels, along with Kenneth Thomson, Olive Tell and Lowell Sherman, who reprised the role he had created in
the original Broadway play.

 This film was a remake of the 1923 silent film version of the same name, produced by Famous Players-Lasky Corporation
</item 4>

<item 5>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 5>

<item 6>
Jonathan Kaplan
Jonathan Kaplan (born November 25, 1947) is an American film producer and director.

 His film "The Accused" (1988) earned actress Jodie Foster her first Oscar for Best Actress and was nominated for the Golden Bear at the 39th Berlin
International Film Festival.

 Kaplan received five Emmy nominations for his roles directing and producing "ER".
</item 6>

<item 7>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 7>

<item 8>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 8>

<item 9>
Richard Sherman (MP)
Richard Sherman ("fl."

 1364-1397) was an English ironmonger and property owner in Derby, who served two terms as a bailiff and served two terms as a Member of Parliament from Derby,
being chosen first in November 1384 (serving with John de Stockes) and again in 1391 (with Thomas Docking).
</item 9>

<item 10>
High Stakes (1931 film)
High Stakes is a 1931 American Pre-Code comedy drama produced and released by RKO Pictures.

 The picture was directed by Lowell Sherman who also stars and marks the last starring screen appearance of silent screen diva Mae Murray.

 It is based on a 1924 Broadway play that starred Sherman playing the same role he plays in this film.
</item 10>
=== END ITEM id=6a1ea3bc696d1811 ===

=== ITEM id=91cf51774ec0d57e ===
Gold sentences and facts:
Sentence 0: Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal Navy surface vessels and submarines.
  Fact 0.0: Rosyth Dockyard is a large naval dockyard
  Fact 0.1: Rosyth Dockyard is on the Firth of Forth
  Fact 0.2: Rosyth Dockyard is at Rosyth, Fife, Scotland
  Fact 0.3: Rosyth Dockyard is owned by Babcock Marine
  Fact 0.4: Rosyth Dockyard formerly undertook refitting of Royal Navy surface vessels
  Fact 0.5: Rosyth Dockyard formerly undertook refitting of Royal Navy submarines
Sentence 1: The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.
  Fact 1.0: The Queen Elizabeth class is a class of two aircraft carriers
  Fact 1.1: The Queen Elizabeth class aircraft carriers belong to the United Kingdom's Royal Navy

Retrieved items:
<item 1>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 1>

<item 2>
Queen Elizabeth-class aircraft carrier
The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.

 The first,  , was named on 4 July 2014, with her ship commissioning planned for 2017, and an initial operating capability expected in 2018.

 The second, HMS "Prince of Wales" , is scheduled to be launched around Summer 2017, followed by commissioning in 2020 and service thereafter.

 On 5 September 2014, at the NATO 2014 Wales summit, the Prime Minister David Cameron announced that the second carrier would be brought into service, ending
years of uncertainty surrounding its future.

 This was confirmed by the November 2015 Government Strategic Defence Review, with both carriers entering service, one being available at any time.
</item 2>

<item 3>
Rosyth Dockyard
Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal
Navy surface vessels and submarines.

 Before its privatisation in the 1990s it was formally the Royal Naval Dockyard Rosyth.

 Its primary role is now as integration site for the Royal Navy's newest aircraft carriers - the "Queen Elizabeth"-class.
</item 3>

<item 4>
Goliath (Rosyth)
Goliath is a crane in Rosyth Dockyard, Scotland, with a lift capacity of 1000 t , the largest in Britain.
</item 4>

<item 5>
List of sunken aircraft carriers
With the advent of heavier-than-air flight, the aircraft carrier has become a decisive weapon at sea.

 In 1911 aircraft began to be successfully launched and landed on ships with the successful flight of a Curtiss Pusher aboard the USS "Pennsylvania".

 The British Royal Navy pioneered the first aircraft carrier as floatplanes, as flying boats under performed compared to traditional land based aircraft.

 The first true aircraft carrier was the HMS "Argus" , launched in late 1917 with a complement of 20 aircraft, a flight deck of 550 ft and 68 ft wide.

 The last aircraft carrier sunk in wartime was the Japanese carrier "Amagi", in Kure Harbour in July 1945.

 The greatest loss of life was the 2,046 killed on the "Akitsu Maru"—a converted passenger liner with a small flight deck, carrying the Imperial Japanese Army's
64th Infantry Regiment.
</item 5>

<item 6>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 6>

<item 7>
HMS Prince of Wales (R09)
HMS "Prince of Wales" is the second "Queen Elizabeth"-class aircraft carrier under construction for the Royal Navy, with plans for active service from 2020.

 She is the seventh Royal Navy ship to have the name HMS "Prince of Wales".

 Construction of the ship began in 2011 at Rosyth Dockyard and in April 2016, was said to be around 80% structurally complete.
</item 7>

<item 8>
Rosyth railway station
Rosyth railway station serves the town of Rosyth in Fife, Scotland.

 The station is managed by Abellio ScotRail and lies on the Fife Circle Line, 23.6 km north of Edinburgh Waverley .

 It was opened in 1917 by the North British Railway (as "Rosyth Halt") to serve the nearby naval dockyard.
</item 8>

<item 9>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 9>

<item 10>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 10>
=== END ITEM id=91cf51774ec0d57e ===

=== ITEM id=4a7d6f1b354b4c58 ===
Gold sentences and facts:
Sentence 0: Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
  Fact 0.0: Marc Rosset was born on 7 November 1970.
  Fact 0.1: Marc Rosset is a former professional tennis player.
  Fact 0.2: Marc Rosset is from Switzerland.
  Fact 0.3: Marc Rosset is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
Sentence 1: Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional tennis player.
  Fact 1.0: Maksim Mirnyi, also known as Max Mirnyi, was born on 6 July 1977 in Minsk.
  Fact 1.1: Maksim Mirnyi's Belarusian name is Максім Мікалаевіч Мірны.
  Fact 1.2: Maksim Mirnyi's Russian name is Максим Николаевич Мирный.
  Fact 1.3: Maksim Mirnyi is a Belarusian professional tennis player.

Retrieved items:
<item 1>
Max Mirnyi
Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional
tennis player.
</item 1>

<item 2>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 2>

<item 3>
The material compiles tennis results and player biographies. Sherwood Stewart and Ferdi Taygan won the 1982 French Open doubles and finished runner-up at the
Volvo Masters and U.S. Pro Indoor; Stewart later partnered others. Jorge Lozano, a Mexican doubles No. 4, won nine doubles titles, including two French Open
mixed doubles crowns, and captained Mexico’s Davis Cup team. Brazilian Carlos Fernandes reached the 1961 French Open quarterfinals. Argentine Javier Fr
</item 3>

<item 4>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 4>

<item 5>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 5>

<item 6>
2010 Australian Open – Mixed Doubles
Sania Mirza and Mahesh Bhupathi were the defending champions.

 They accepted the invitation for the tournament, but withdrew before their match against Carly Gullickson and Bernard Tomic and were replaced by Akgul
Amanmuradova and Rik de Voest.
</item 6>

<item 7>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 7>

<item 8>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 8>

<item 9>
In 2009, Roger Federer won the French Open and Wimbledon, completed the Career Grand Slam, and claimed a record 15th major; he also won Masters titles in Madrid
and Cincinnati. Hicham Arazi, a Moroccan player ranked as high as No. 22, won Casablanca and reached four major quarterfinals. Swiss Marc Rosset won Olympic
singles gold in 1992, the French Open doubles title, and the 2000 AXA Cup. Nag
</item 9>

<item 10>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 10>
=== END ITEM id=4a7d6f1b354b4c58 ===

=== ITEM id=495d9e1381baa830 ===
Gold sentences and facts:
Sentence 0: Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.
  Fact 0.0: Hou Hsiao-hsien was born on 8 April 1947.
  Fact 0.1: Hou Hsiao-hsien is a Taiwanese film director.
  Fact 0.2: Hou Hsiao-hsien is a Taiwanese screenwriter.
  Fact 0.3: Hou Hsiao-hsien is a Taiwanese producer.
  Fact 0.4: Hou Hsiao-hsien is a Taiwanese actor.
Sentence 1:  He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.
  Fact 1.0: Hou Hsiao-hsien is a leading figure in world cinema.
  Fact 1.1: Hou Hsiao-hsien is a leading figure in Taiwan's New Wave cinema movement.
  Fact 1.2: Hou Hsiao-hsien is an auteur.
Sentence 2:  He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in 2015 for "The Assassin" (2015).
  Fact 2.0: Hou Hsiao-hsien won the Golden Lion at the Venice Film Festival in 1989.
  Fact 2.1: Hou Hsiao-hsien won the Golden Lion for his film "A City of Sadness" (1989).
  Fact 2.2: Hou Hsiao-hsien won the Best Director award at the Cannes Film Festival in 2015.
  Fact 2.3: Hou Hsiao-hsien won the Best Director award at Cannes for his film "The Assassin" (2015).
Sentence 3:  Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
  Fact 3.0: "The Puppetmaster" (1993) is a highly regarded work by Hou Hsiao-hsien.
  Fact 3.1: "Flowers of Shanghai" (1998) is a highly regarded work by Hou Hsiao-hsien.
Sentence 4: Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.
  Fact 4.0: Kenny Ortega (Kenneth John Ortega) was born on April 18, 1950.
  Fact 4.1: Kenny Ortega is an American producer.
  Fact 4.2: Kenny Ortega is an American director.
  Fact 4.3: Kenny Ortega is an American choreographer.
Sentence 5:  He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
  Fact 5.0: Kenny Ortega is best known for directing "Hocus Pocus".
  Fact 5.1: Kenny Ortega is best known for directing the "High School Musical" trilogy.
  Fact 5.2: Kenny Ortega is best known for directing "Descendants".
  Fact 5.3: Kenny Ortega is best known for directing Michael Jackson's "This Is It" concerts.

Retrieved items:
<item 1>
The Assassin (2015 film)
The Assassin (; or: "The Assassin Niè Yǐnniáng") is a 2015 wuxia film directed by Taiwanese director Hou Hsiao-hsien.

 A Taiwan/China/Hong Kong co-production, it was an official selection in the main competition section at the 2015 Cannes Film Festival.

 At Cannes, Hou won the award for Best Director.

 It was released in China and Hong Kong on 27 August, and a day later in Taiwan on 28 August 2015.

 It was selected as the Taiwanese entry for the Best Foreign Language Film at the 88th Academy Awards but it was not nominated.
</item 1>

<item 2>
Academy Honorary Award
The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards (previously called the Special Award, which was first presented in early 1929) – is
given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not
covered by existing Academy Awards, although prior winners of competitive Academy Awards are not excluded from receiving the Honorary Award.

 Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.

 Unlike the Special Achievement Award instituted in 1972 (and discontinued in 1995), those on whom the Academy confers its Honorary Award do not have to meet
"the Academy's eligibility year and deadline requirements."

 Like the Special Achievement Award, the Special Award and Honorary Award have been used to reward significant achievements of the year that did not fit in
existing categories, subsequently leading the Academy to establish several new categories, and to honor exceptional career achievements, contributions to the
motion picture industry, and service to the Academy.
</item 2>

<item 3>
Academy Award for Best Film Editing
This process is essentially the reverse of that of the British Academy of Film and Television Arts (BAFTA); nominations for the BAFTA Award for Best Editing are
done by a general ballot of Academy voters, and the winner is selected by members of the editing chapter.
</item 3>

<item 4>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 4>

<item 5>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 5>

<item 6>
Kenny Ortega
Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.

 He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
</item 6>

<item 7>
Hou Hsiao-hsien
Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.

 He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.

 He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in
2015 for "The Assassin" (2015).

 Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
</item 7>

<item 8>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 8>

<item 9>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 9>

<item 10>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 10>
=== END ITEM id=495d9e1381baa830 ===

=== ITEM id=5ed49d707ae49f50 ===
Gold sentences and facts:
Sentence 0: Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.
  Fact 0.0: Underdog is a 2007 American film.
  Fact 0.1: Underdog is a family superhero comedy film.
  Fact 0.2: Underdog is based on the 1960s cartoon series of the same name.
Sentence 1: Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi, a Life in the Woods" by Austrian author Felix Salten.
  Fact 1.0: Bambi is a 1942 American animated film.
  Fact 1.1: Bambi was directed by David Hand.
  Fact 1.2: David Hand supervised a team of sequence directors on Bambi.
  Fact 1.3: Bambi was produced by Walt Disney.
  Fact 1.4: Bambi is based on the book 'Bambi, a Life in the Woods'.
  Fact 1.5: The book 'Bambi, a Life in the Woods' was written by Austrian author Felix Salten.

Retrieved items:
<item 1>
Bambi
Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Walt Disney and based on the book "Bambi,
a Life in the Woods" by Austrian author Felix Salten.

 The film was released by RKO Radio Pictures on August 13, 1942, and is the fifth Disney animated feature film.
</item 1>

<item 2>
Bosko the Musketeer
Bosko the Musketeer is an American animated short film.

 It is a "Looney Tunes" cartoon, featuring Bosko, the first star of the series.

 It was released on August 12, 1933, although some sources note September 16 as a date; this is problematic, as that would imply that the last films featuring
Bosko as the star of Warner Bros. cartoons were released after the first film featuring Buddy, the second star of the series.

 It was, like most "Looney Tunes" of the time, directed by Hugh Harman; Frank Marsales was the musical director.
</item 2>

<item 3>
Underdog (film)
Underdog is a 2007 American family superhero comedy film based on the 1960s cartoon series of the same name.

 Directed by Frederik Du Chau and written by Joe Piscatella, Adam Rifkin, and Craig A. Williams, the film stars Jim Belushi, Peter Dinklage, John Slattery, and
Patrick Warburton with the voice talents of Jason Lee, Amy Adams, and Brad Garrett.

 Unlike the TV series, the Underdog character is portrayed as a regular dog rather than an anthropomorphic one.

 Underdog, voiced by Jason Lee, was played by a lemon beagle named Leo sporting a red sweater and a blue cape.

 The film grossed $65.3 million worldwide.
</item 3>

<item 4>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 4>

<item 5>
Roadside Romeo
Roadside Romeo is a 2008 3D Indian-American computer animated romantic musical comedy family film written and directed by Jugal Hansraj and produced by Aditya
Chopra and Yash Chopra of Yash Raj Films and distributed by Walt Disney Studios Motion Pictures in United States, United Kingdom and India.

 It was released on 24 October 2008 in the United States and India.

 An Arabic dub was released in Kuwait on October 23.

 This was the second Bollywood movie to receive a North American release by a Hollywood studio, following Sony Pictures' "Saawariya" (2007).

 The title character is a dog living in Mumbai, as voiced by Saif Ali Khan; his girlfriend, Laila, is voiced by Kareena Kapoor.

 This was the first voice-over in an animated production for both actors.

 "Roadside Romeo" was also Hansraj's directorial debut.

 "Roadside Romeo" received generally negative reviews from critics, with most of the criticism focused on the film's script, predictable plot and overuse of
cliches.
</item 5>

<item 6>
Buddy Baker (composer)
Norman Dale "Buddy" Baker (January 4, 1918 – July 26, 2002) was an American composer who, together with Paul J. Smith, scored many Disney films, such as "The
Apple Dumpling Gang" in 1975, "The Apple Dumpling Gang Rides Again" in 1979, "The Shaggy D.A." in 1976, "The Many Adventures of Winnie the Pooh" in 1977, and
"The Fox and the Hound" in 1981.
</item 6>

<item 7>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 7>

<item 8>
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 8>

<item 9>
Batman vs. Two-Face
Batman vs. Two-Face is an upcoming 2017 American animated direct-to-video superhero film produced by Warner Bros.

 Animation and distributed by Warner Bros.

 Home Entertainment.

 It is a sequel to "".

 It will premiere at the New York Comic Con on October 8, 2017 and will be followed by a digital release on October 10, 2017 and on DVD and Blu-ray on October
17, 2017.

 Based on the 1960s "Batman" TV series, the film stars Adam West (in his final role before his death), Burt Ward and Julie Newmar reprising their roles of
Batman, Robin and Catwoman from the series.

 The film will show a tribute to the late Adam West.

 This could also be the final time Warner Bros.

 Animation would make a 1960s Batman film.
</item 9>

<item 10>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 10>
=== END ITEM id=5ed49d707ae49f50 ===

=== ITEM id=4dea6aa0682bce4b ===
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
A botanical survey covers African Huernia, whose carrion-scented flowers attract flies; Asian water vegetable Oenanthe javanica; and many plant genera,
families, and species, with California botanist Samuel Parish commemorated in plant names. Nepenthes comprises about 150 Old World tropical pitcher plants,
especially diverse in Borneo, Sumatra, and the Philippines; monkeys drink pitcher rainwater, and several Bornean species and hybrids are noted. Other highlights
include a
</item 1>

<item 2>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 2>

<item 3>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 3>

<item 4>
Nepenthes boschiana
Nepenthes boschiana ( ; after Johannes van den Bosch), or Bosch's Pitcher-Plant, is a tropical pitcher plant endemic to Borneo.

 It is most closely allied to "N. faizaliana".

 "Nepenthes borneensis" is considered a synonym of this species.

 "Nepenthes boschiana" has no known natural hybrids.

 No valid forms or varieties have been described.

 "Nepenthes boschiana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. chaniana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 4>

<item 5>
Nepenthes tenax
Nepenthes tenax ( ; from Latin: "tenax" "tenacious") is a lowland species of tropical pitcher plant native to northern Queensland, Australia.

 It is the third "Nepenthes" species recorded from the continent and its second endemic species.

 "Nepenthes tenax" is closely related to the two other Australian "Nepenthes" species: "N. mirabilis" and "N. rowaniae".
</item 5>

<item 6>
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
</item 6>

<item 7>
Nepenthes is a genus of roughly 150 Old World tropical pitcher-plant species, richest in Borneo, Sumatra, and the Philippines; monkeys sometimes drink rainwater
from its pitchers. Bornean N. boschiana (with N. borneensis a synonym) is allied to N. faizaliana; N. chaniana has long white hairs and pale cylindrical
pitchers. N. × kinabaluensis, a cross of N. rajah and N. vil
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
Nepenthes veitchii
Nepenthes veitchii ( or ; after James Veitch, nurseryman of the Veitch Nurseries), or Veitch's Pitcher-Plant, is a "Nepenthes" species from the island of
Borneo.

 The plant is widespread in north-western Borneo and can also be found in parts of Kalimantan.

 "Nepenthes veitchii" usually grows as an epiphyte, though the form from Bario seems to be strictly terrestrial and has not been observed to climb trees.
</item 9>

<item 10>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 10>
=== END ITEM id=4dea6aa0682bce4b ===

=== ITEM id=c0374516833d00c8 ===
Gold sentences and facts:
Sentence 0: Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.
  Fact 0.0: Neon City is also known as Anno 2053 in Italy.
  Fact 0.1: Neon City is also known as Neonski Grad in Serbia.
  Fact 0.2: Neon City is a 1991 Canadian film.
  Fact 0.3: Neon City is a post-apocalyptic science fiction film.
  Fact 0.4: Neon City was directed by Monte Markham.
  Fact 0.5: Neon City was written by Jeff Begun.
  Fact 0.6: Neon City was written by Ann Lewis Hamilton.
  Fact 0.7: Neon City's writing was credited under the pseudonym Buck Finch.
Sentence 1:  The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
  Fact 1.0: Michael Ironside is in the cast of Neon City.
  Fact 1.1: Vanity is in the cast of Neon City.
  Fact 1.2: Lyle Alzado is in the cast of Neon City.
  Fact 1.3: Valerie Wildman is in the cast of Neon City.
  Fact 1.4: Nick Klar is in the cast of Neon City.
  Fact 1.5: Juliet Landau is in the cast of Neon City.
  Fact 1.6: Richard Sanders is in the cast of Neon City.
  Fact 1.7: Monte Markham is in the cast of Neon City.
Sentence 2: Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and intimidating style of play.
  Fact 2.0: Lyle Martin Alzado was born on April 3, 1949.
  Fact 2.1: Lyle Martin Alzado died on May 14, 1992.
  Fact 2.2: Lyle Martin Alzado was a professional American football defensive end.
  Fact 2.3: Lyle Martin Alzado played in the National Football League.
  Fact 2.4: Lyle Martin Alzado was famous for his intense and intimidating style of play.

Retrieved items:
<item 1>
Alec Gibson
Alec Raymond Gibson (born December 9, 1963 in Columbus, Ohio) is a former American football defensive end in the National Football League for the Washington
Redskins.

 Gibson played in three games for the Washington Redskins in 1987.

 The Washington Redskins went 3-0 during the strike.

 Defeating the St. Louis Cardinals, New York Giants, and Dallas Cowboys.

 Gibson caused Hall of Famer Tony Dorsett to fumble the football twice on Monday Night Football in Dallas the last replacement game played in 1987.

 He played college football at Ventura Junior College where he was an All-American defensive tackle and at the University of Illinois as a defensive end.

 In 1988, he played Arena football for the New York Knights.

 Gibson played 11 of 12 games before injuring his right knee (ACL) which put an end to his career.
</item 1>

<item 2>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 2>

<item 3>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 3>

<item 4>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 4>

<item 5>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 5>

<item 6>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 6>

<item 7>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 7>

<item 8>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 8>

<item 9>
Batman (1989 film)
Batman is a 1989 American superhero film directed by Tim Burton and produced by Jon Peters and Peter Guber, based on the DC Comics character of the same name.

 It is the first installment of Warner Bros.' initial "Batman" film series.

 The film stars Michael Keaton as Bruce Wayne/Batman and Jack Nicholson as The Joker, alongside Kim Basinger, Robert Wuhl, Pat Hingle, Billy Dee Williams,
Michael Gough and Jack Palance.

 The film takes place early in the title character's war on crime, and depicts a battle with his arch-nemesis the Joker.
</item 9>

<item 10>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 10>
=== END ITEM id=c0374516833d00c8 ===
