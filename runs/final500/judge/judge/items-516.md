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

=== ITEM id=14041b47fbea719e ===
Gold sentences and facts:
Sentence 0: Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.
  Fact 0.0: The Dogo Cubano is also known as the Cuban Mastiff.
  Fact 0.1: The Dogo Cubano is also known as the Cuban Dogo.
  Fact 0.2: The Dogo Cubano is also known as the Cuban Dogge.
  Fact 0.3: The Dogo Cubano is an extinct dog breed.
  Fact 0.4: The Dogo Cubano is a dog breed from Cuba.
Sentence 1: The Dutch Shepherd is a herding dog of Dutch origin.
  Fact 1.0: The Dutch Shepherd is a herding dog.
  Fact 1.1: The Dutch Shepherd is a herding dog of Dutch origin.

Retrieved items:
<item 1>
Dogo Cubano
Dogo Cubano also known as the Cuban Mastiff, Cuban Dogo and Cuban Dogge is an extinct dog breed from Cuba.

 It was of the Bull Mastiff type and was used for dog fighting.
</item 1>

<item 2>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 2>

<item 3>
Cuban Criollo
The Criollo Cubano originates from Spanish horses brought to Cuba by Diego Velázquez in 1751.

 There are four different breeds known as criollo: the Cubano de Paso, the Pinto Cubano, the Criollo de Trote and the Patibarcino.

 It is a small stout breed used by the Guajiro people.
</item 3>

<item 4>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 4>

<item 5>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 5>

<item 6>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 6>

<item 7>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 7>

<item 8>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 8>

<item 9>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 9>

<item 10>
The Bulldog is a muscular, wrinkled English breed, ranked fourth among US purebreds in 2007. West Siberian Laikas are versatile barking hunting dogs whose
formal breeding resumed in 1946. Mediterranean warren hounds, including Cretan and Canary Island hounds, typically hunt using smell, sight, and hearing. Segugio
Italiano has short- and wire-haired varieties. Guatemala’s national Dogo is locally recognized; Cuba’s Bull Mastiff-type Dog
</item 10>
=== END ITEM id=14041b47fbea719e ===

=== ITEM id=8fc95c733c9ab9a2 ===
Gold sentences and facts:
Sentence 0: Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.
  Fact 0.0: Xingcheng is a county-level city of southwest Liaoning province, China.
  Fact 0.1: Xingcheng's former name was Ningyuan (宁远).
  Fact 0.2: Xingcheng has a population of approximately 140,000 urban inhabitants.
  Fact 0.3: Xingcheng is located on Liaodong Bay, the northern coast of the Bohai Sea.
Sentence 1:  It covers an area of 2046 square kilometers and has a population of about one million.
  Fact 1.0: Xingcheng covers an area of 2046 square kilometers.
  Fact 1.1: Xingcheng has a population of about one million.

Retrieved items:
<item 1>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 1>

<item 2>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 2>

<item 3>
Huan River
The Huan River (), or Anyang River (安阳河), is a river in Henan, China, and part of the Hai River basin.

 The river rises north of Linzhou in northwestern Henan, and joins the Wei River near Neihuang in the northeast of the province.
</item 3>

<item 4>
Shanxi–Henan–Shandong Railway
The Shanxi–Henan–Shandong Railway () or the Jinyulu Railway, also known as the Shanxi South Central Railway or the Watang–Rizhao Railway, is an electrified
double-track heavy freight railway across northern China.

 The line runs 1260 km from the village of Watang in Lüliang, Shanxi Province to the port of Rizhao in Shandong Province, and is named after the three provinces
through which it passes Jin (Shanxi), Yu (Henan) and Lu (Shandong).

 The line is the world's longest heavy freight railway and is designed to facilitate the export of coal from Shanxi to overseas markets.

 The line can accommodate train speeds of up to 120 km/h .
</item 4>

<item 5>
Shanxi–Henan–Shandong Railway
The line can accommodate train speeds of up to 120 km/h .

 Cities along route include Lüliang, Liulin, Hongdong, Changzhi and Pingshun in Shanxi; Linzhou, Anyang, Tangyin, Puyang, Hebi, Fan County and Taiqian County in
Henan; Liangshan County, Dongping, Ningyang, Xintai, Laiwu, Yiyuan, Yishui, Ju County, Junan and Rizhao in Shandong.

 The line was built from 2010 to 2014 and operations began on December 30, 2014.
</item 5>

<item 6>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 6>

<item 7>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 7>

<item 8>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 8>

<item 9>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 9>

<item 10>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 10>
=== END ITEM id=8fc95c733c9ab9a2 ===

=== ITEM id=7d6b79f35a68acdf ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Cage starred in a variety of films.
  Fact 0.1: Cage starred in Valley Girl, released in 1983.
  Fact 0.2: Cage starred in Racing with the Moon, released in 1984.
  Fact 0.3: Cage starred in Birdy, released in 1984.
  Fact 0.4: Cage starred in Peggy Sue Got Married, released in 1986.
  Fact 0.5: Cage starred in Raising Arizona, released in 1987.
  Fact 0.6: Cage starred in Moonstruck, released in 1987.
  Fact 0.7: Cage starred in Vampire's Kiss, released in 1989.
  Fact 0.8: Cage starred in Wild at Heart, released in 1990.
  Fact 0.9: Cage starred in Fire Birds, released in 1990.
  Fact 0.10: Cage starred in Honeymoon in Vegas, released in 1992.
  Fact 0.11: Cage starred in Red Rock West, released in 1993.
Sentence 1: Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.
  Fact 1.0: Moonstruck is a 1987 American film.
  Fact 1.1: Moonstruck is a romantic comedy film.
  Fact 1.2: Moonstruck was directed by Norman Jewison.
  Fact 1.3: Moonstruck was written by John Patrick Shanley.
Sentence 2:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 2.0: Moonstruck is about a widowed 37-year-old Italian-American woman played by Cher.
  Fact 2.1: The woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 2.2: The fiancé is played by Danny Aiello.
  Fact 2.3: The younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 1>

<item 2>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 2>

<item 3>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 3>

<item 4>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 4>

<item 5>
Dying of the Light (film)
Dying of the Light is a 2014 American psychological thriller film written and directed by Paul Schrader and starring Nicolas Cage, Anton Yelchin and Irène Jacob
about a government agent who must track down and kill a terrorist before he loses his full memory from a disease.

 It was released theatrically and through VOD formats by Lionsgate on December 5, 2014.

 The film received extremely negative reviews, with controversy surrounding the heavy tampering and reediting of the footage by the studio, who denied Schrader
final-cut privilege and led him and principal members of the cast to disown the released version and campaign against it.
</item 5>

<item 6>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 6>

<item 7>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 7>

<item 8>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 8>

<item 9>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 9>

<item 10>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 10>
=== END ITEM id=7d6b79f35a68acdf ===

=== ITEM id=fa9a16aedb280b11 ===
Gold sentences and facts:
Sentence 0:  It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center near the Miami River.
  Fact 0.0: It flows in a west to east direction.
  Fact 0.1: It flows from the western part of the state in the Everglades.
  Fact 0.2: It flows past the Miami International Airport.
  Fact 0.3: It flows to a salinity control center near the Miami River.
Sentence 1: The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
  Fact 1.0: The Genesee River is a tributary of Lake Ontario.
  Fact 1.1: The Genesee River flows northward.
  Fact 1.2: The Genesee River flows through the Twin Tiers of Pennsylvania.
  Fact 1.3: The Genesee River flows through the Twin Tiers of New York in the United States.

Retrieved items:
<item 1>
Genesee River
The Genesee River is a tributary of Lake Ontario flowing northward through the Twin Tiers of Pennsylvania and New York in the United States.
</item 1>

<item 2>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 2>

<item 3>
Graph query row:
waterway: Tamiami Canal
relationship: LOCATED_IN
connected_place: Florida
place_kind: state
country: United States
</item 3>

<item 4>
Graph query row:
waterway: Genesee River
relationship: LOCATED_IN
connected_place: Pennsylvania
place_kind: state
country: United States
</item 4>

<item 5>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 5>

<item 6>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 6>

<item 7>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 7>

<item 8>
Graph query row:
waterway: Genesee River
relationship: LOCATED_IN
connected_place: New York
place_kind: state
country: United States
</item 8>

<item 9>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 9>

<item 10>
Genesee Valley Park
Genesee Valley Park is located in the south side of Rochester, New York along the shores of the Genesee River.

 The New York State Barge Canal (the currently in-use portion of the Erie Canal) crosses the Genesee River within the park.

 The University of Rochester is located near the park's north entrance.
</item 10>
=== END ITEM id=fa9a16aedb280b11 ===

=== ITEM id=ddaa89446c8ea1a8 ===
Gold sentences and facts:
Sentence 0:  The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
  Fact 0.0: The area along Northern Boulevard is well known for its high-end premium open-air shopping center.
  Fact 0.1: The high-end premium open-air shopping center is named the Americana Manhasset.
Sentence 1: International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van Sweden.
  Fact 1.0: Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine, Tiffany & Co., and Prada are international fashion designer labels and luxury goods companies.
  Fact 1.1: These companies maintain a total of more than 60 retail stores and restaurants.
  Fact 1.2: These stores and restaurants are on a 220000 sqft site.
  Fact 1.3: The 220000 sqft site is surrounded by gardens.
  Fact 1.4: The gardens surrounding the site were designed by Oheme van Sweden.

Retrieved items:
<item 1>
Daimler, Stuttgart-based German automaker, owned numerous vehicle brands and sold 3 million vehicles in 2016; it was the world’s largest truck maker. Estate
jewelry means significant antique, vintage, or pre-owned jewelry. Mouawad is a Geneva- and Dubai-based luxury jeweler founded in Beirut in 1891. Dior is a
French luxury group controlled by Bernard Arnault and holds a major stake in LVMH. Concord is a Swiss watch brand now owned by Mov
</item 1>

<item 2>
Bumble and bumble
Bumble and bumble Products, LLC is one of 27 brands owned by Estée Lauder Companies, Inc.

 It was founded as a New York City hair salon in 1977 by hairdresser and entrepreneur Michael Gordon.

 The company operates two flagship salons in Manhattan: a midtown location on East 56th Street and on the top two floors of The House of Bumble in the
Meatpacking District.
</item 2>

<item 3>
Daimler, once the world’s largest truckmaker, sold 3 million vehicles in 2016. Mouawad, founded in Beirut in 1891, is a Geneva/Dubai jeweler; Dior is controlled
by Bernard Arnault and holds a major LVMH stake. Microsoft My Phone backed up Windows Mobile data (2009–11). Samsung products ranged from budget phones to
Galaxy flagships and tablets. SAIC-GM-Wuling sold 1.445 million vehicles
</item 3>

<item 4>
Miracle Mile (Manhasset)
The Miracle Mile is a prominent shopping district in Manhasset on the North Shore of Long Island in Nassau County, New York.

 The area along Northern Boulevard is well known for its high-end premium open-air shopping center, the Americana Manhasset.
</item 4>

<item 5>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 5>

<item 6>
Americana Manhasset
International fashion designer labels and luxury goods companies—Hermes, Dior, Cartier, Bottega Veneta, Chanel, Fendi, Gucci, Louis Vuitton, MaxMara, Celine,
Tiffany & Co., and Prada—maintain a total of more than 60 retail stores and restaurants on a 220000 sqft site surrounded by gardens designed by Oheme van
Sweden.

 Peter Marino is Americana Manhasset's master architect.
</item 6>

<item 7>
Estée Lauder Companies
The Estée Lauder Companies Inc. ( ) is an American manufacturer and marketer of prestige skincare, makeup, fragrance and hair care products.

 The company owns a diverse portfolio of brands, distributed internationally through retail channels and digital commerce, and has its headquarters in Midtown
Manhattan, New York City.
</item 7>

<item 8>
Luxify
Luxify is a Hong Kong-based company, providing an online marketplace for new, vintage and pre-owned luxury goods.

 The company manages an online marketplace website in which members buy and sell a variety of luxury goods.

 It was Hong Kong's first online marketplace to buy and sell luxury goods The site has over 15 categories, including real estate, watches, jewelry, luxury cars,
yachts, aircraft, luxury handbags, art, collectibles, furniture, antiques and fine wines & spirits.

 As of March 2016, the website had over 22,000 listings from luxury dealers from all over the world, making a US$18.6 billion marketplace.

 Luxify also has offices in Singapore and London and local partners in Shanghai and Jakarta.
</item 8>

<item 9>
Mouawad
Mouawad is a privately held Swiss and Emirati luxury goods company that makes fashion accessories, jewelry and retail company that sources diamonds and
gemstones, designs, manufacturers, and sells jewelry collections, objects of art, and luxury watches.

 The firm has headquarters in Geneva, Switzerland, with a Middle East headquarters at Jumeirah Lakes Towers in Dubai.

 Founded in 1891 in Beirut, Lebanon by David Mouawad, the firm is now led by 4th generation co-guardians Fred Mouawad, Alain Mouawad and Pascal Mouawad.

 The firm's jewelry and watch design and manufacturing divisions are located in Switzerland and Asia Pacific.
</item 9>

<item 10>
Christian Dior SE
Christian Dior SE (] ), commonly known as Dior, is a French luxury goods company controlled and chaired by French businessman Bernard Arnault, who also heads
LVMH – the world's largest luxury group.

 Dior itself holds 42.36% shares of and 59.01% voting rights within LVMH.
</item 10>
=== END ITEM id=ddaa89446c8ea1a8 ===

=== ITEM id=5249d3a9b0cfd64e ===
Gold sentences and facts:
Sentence 0:  This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's "Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
  Fact 0.0: This episode is a milestone in the history of the series.
  Fact 0.1: In this episode, Snowball II is killed off.
  Fact 0.2: Homer's mother died in season 19's Mona Leaves-a.
  Fact 0.3: Excluding the death of Homer's mother, the killing of Snowball II is to date the closest thing to an actual Simpson family member being killed off.
Sentence 1: Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".
  Fact 1.0: Mona Penelope Simpson's maiden name is Olsen.
  Fact 1.1: Mona Penelope Simpson is a recurring fictional character.
  Fact 1.2: Mona Penelope Simpson is a character in the animated television series The Simpsons.

Retrieved items:
<item 1>
Homer to the Max
"Homer to the Max" is the thirteenth episode of "The Simpsons"<nowiki>'</nowiki> tenth season.

 It originally aired on the Fox network in the United States on February 7, 1999.

 In the episode, Homer discovers that a new television show, "Police Cops", has a hero also named Homer Simpson.

 He is delighted with the positive attention he receives because of his name, but when the television character is rewritten from a hero to a bumbling idiot, he
is mocked and taunted, so he changes his name to "Max Power" to rid himself of the negative attention.

 Max gains new friends, and is forced into a protest to prevent a forest from being knocked down.

 In the end, he changes his name back to Homer Simpson.
</item 1>

<item 2>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 2>

<item 3>
Girly Edition
"Girly Edition" is the twenty-first episode in the ninth season of the American animated television series "The Simpsons".

 It originally aired on the Fox network in the United States on April 19, 1998.

 In the episode, Lisa and Bart Simpson must co-anchor a new news program, though when Bart is seen as a more successful news anchor, Lisa becomes jealous and
seeks revenge.

 Meanwhile, in the subplot, Homer Simpson gets a monkey helper because of his laziness.

 "Girly Edition" was the first episode written by Larry Doyle and was directed by Mark Kirkland.

 Much of the subplot was inspired by the film "Monkey Shines".

 Critics gave the episode positive reviews and it was well received by Lisa's voice actress Yeardley Smith.
</item 3>

<item 4>
I, (Annoyed Grunt)-bot
"I, (Annoyed Grunt)-Bot", also known as ""I, D'oh-Bot"", is the ninth episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It originally aired on the Fox network in the United States on January 11, 2004.

 This episode represents a milestone in the history of the series as Snowball II is killed off, which, excluding the death of Homer's mother in season 19's
"Mona Leaves-a", is to date the closest thing to an actual Simpson family member actually being killed off.
</item 4>

<item 5>
My Mother the Carjacker
"My Mother the Carjacker" is the second episode of "The Simpsons"<nowiki>'</nowiki> fifteenth season.

 It first aired on the Fox network in the United States on November 9, 2003.

 Homer receives a cryptic message in the newspaper informing him to come to a certain place at midnight, and soon discovers that the person who wrote the
message is his mother, Mona Simpson.

 It was written by Michael Price and directed by Nancy Kruse.

 Glenn Close makes her second of six guest spots as Homer's mother.

 It has a direct link from the season seven episode, "Mother Simpson".

 It was nominated for a Writers Guild of America Award in 2004.

 In its original run, the episode received 12.4 million viewers.
</item 5>

<item 6>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 6>

<item 7>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 7>

<item 8>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 8>

<item 9>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 9>

<item 10>
Grampa Simpson
Abraham Jedediah Simpson II, often known as Grampa, is a fictional character in the animated television series "The Simpsons".

 He made his first appearance in the episode entitled "Grampa and the Kids", a Simpsons short on "The Tracey Ullman Show".

 Voiced by Dan Castellaneta, he is the father of Homer Simpson and the grandfather of Bart, Lisa and Maggie Simpson.

 In the 1000th issue of "Entertainment Weekly", Grampa was selected as the Grandpa for "The Perfect TV Family".
</item 10>
=== END ITEM id=5249d3a9b0cfd64e ===

=== ITEM id=0b2ecb34ca825e34 ===
Gold sentences and facts:
Sentence 0: Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.
  Fact 0.0: Karen Joy Fowler was born on February 7, 1950.
  Fact 0.1: Karen Joy Fowler is American.
  Fact 0.2: Karen Joy Fowler is an author of science fiction.
  Fact 0.3: Karen Joy Fowler is an author of fantasy.
  Fact 0.4: Karen Joy Fowler is an author of literary fiction.
Sentence 1: Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.
  Fact 1.0: Charles Bruce Chatwin was born on 13 May 1940.
  Fact 1.1: Charles Bruce Chatwin died on 18 January 1989.
  Fact 1.2: Charles Bruce Chatwin was English.
  Fact 1.3: Charles Bruce Chatwin was a travel writer.
  Fact 1.4: Charles Bruce Chatwin was a novelist.
  Fact 1.5: Charles Bruce Chatwin was a journalist.

Retrieved items:
<item 1>
Bruce Chatwin
Charles Bruce Chatwin (13 May 194018 January 1989) was an English travel writer, novelist, and journalist.

 His first book, "In Patagonia" (1977), established Chatwin as a travel writer, although he considered himself instead a storyteller, interested in bringing to
light unusual tales.

 He won the James Tait Black Memorial Prize for his novel "On the Black Hill" (1982) and his novel "Utz" (1988) was shortlisted for the Man Booker Prize.

 In 2008 "The Times" named Chatwin #46 on their list of "50 Greatest British Writers Since 1945."
</item 1>

<item 2>
Graph query row:
person: Bruce Chatwin
birth_date: 13 May 1940
</item 2>

<item 3>
The Jane Austen Book Club
The Jane Austen Book Club is a 2004 novel by American author Karen Joy Fowler.

 The story, which takes place near Sacramento, California, centers on a book club consisting of five women and one man who meet once a month to discuss Jane
Austen's six novels.

 It was a critical success and became a national bestseller.
</item 3>

<item 4>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 4>

<item 5>
Graph query row:
person: Karen Joy Fowler
birth_date: February 7, 1950
</item 5>

<item 6>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 6>

<item 7>
We Are All Completely Beside Ourselves
We Are All Completely Beside Ourselves is a 2013 novel by the American writer Karen Joy Fowler.

 The novel won the 2014 PEN/Faulkner Award for Fiction<ref name="The Guardian - 3 April 2014 - Karen Joy Fowler wins PEN/Faulkner award for fiction"> </ref> and
was also short-listed for the 2014 Man Booker Prize.
</item 7>

<item 8>
Karen Joy Fowler
Karen Joy Fowler (born February 7, 1950) is an American author of science fiction, fantasy, and literary fiction.

 Her work often centers on the nineteenth century, the lives of women, and alienation.
</item 8>

<item 9>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 9>

<item 10>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 10>
=== END ITEM id=0b2ecb34ca825e34 ===

=== ITEM id=143e2de2c16df8d9 ===
Gold sentences and facts:
Sentence 0: Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.
  Fact 0.0: Corn Ranch is a spaceport.
  Fact 0.1: Corn Ranch is located in the West Texas town of Van Horn, Texas.
  Fact 0.2: Flight tests of the New Shepard are carried out at Corn Ranch.
  Fact 0.3: Flight tests of the New Shepard are carried out by Blue Origin.
Sentence 1: Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its headquarters in Kent, Washington.
  Fact 1.0: Blue Origin is an American privately funded aerospace manufacturer.
  Fact 1.1: Blue Origin is a spaceflight services company.
  Fact 1.2: Blue Origin was set up by Amazon.com founder Jeff Bezos.
  Fact 1.3: Blue Origin has its headquarters in Kent, Washington.

Retrieved items:
<item 1>
Corn Ranch
Corn Ranch is a spaceport in the West Texas town of Van Horn, Texas, where flight tests of the New Shepard are carried out by Blue Origin.

 The 165,000-acre (670 km²) land parcel was purchased by Internet billionaire Jeff Bezos.

 The first flight test took place on November 13, 2006 with the goal of providing commercial tourist flights.
</item 1>

<item 2>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 2>

<item 3>
Northern California TRACON
Northern California TRACON (NCT) (Terminal Radar Approach Control), or NorCal TRACON for short, is an air traffic control facility that provides safety alerts,
separation, and sequencing of air traffic arriving, departing, and transiting the airspace and airports in Northern California.

 Located in Rancho Cordova near Sacramento, NCT controls airspace over 19000 square miles, and serves Reno International Airport, Sacramento International
Airport, San Jose International Airport, Oakland International Airport, and San Francisco International Airport, plus 19 other smaller airports with air traffic
control towers.

 NCT is the 3rd busiest TRACON in America.

 NorCal TRACON is the step between local control (in an airport's control tower) and Air Route Traffic Control Center (ARTCC), in this case, Oakland Center
(ICAO code: ZOA).

 San Francisco International Airport is the 2nd largest airport in California and the largest airport serving Northern California.
</item 3>

<item 4>
SyberJet Aircraft
SyberJet Aircraft is the manufacturer of the SJ30 light business jet.

 The company's headquarters is in Cedar City, Utah adjacent to the Cedar City Regional Airport with additional engineering offices and manufacturing, service,
repair and fatigue test facilities near and on the San Antonio International Airport in San Antonio, Texas.
</item 4>

<item 5>
LSO (company)
LSO, formerly Lone Star Overnight, is a regional shipping carrier that focuses on overnight delivery, utilizing both air and ground transportation, to every
address in Texas, southeastern New Mexico and all major metro markets in Oklahoma, Louisiana, Arkansas, Alabama & Tennessee, plus the country of Mexico.

 Headquartered in Austin, Texas, LSO offers hundreds of drop box locations throughout its service area.

 Like its competitors, LSO’s brand distinguishes itself with a signature shade of blue.
</item 5>

<item 6>
Blue Origin
Blue Origin is an American privately funded aerospace manufacturer and spaceflight services company set up by Amazon.com founder Jeff Bezos with its
headquarters in Kent, Washington.

 The company is developing technologies to enable private human access to space with the goal to dramatically lower costs and increase reliability.

 Blue Origin is employing an incremental approach from suborbital to orbital flight, with each developmental step building on its prior work.

 The company motto is ""Gradatim Ferociter"", Latin for "Step by Step, Ferociously".

 Blue Origin is developing a variety of technologies, with a focus on rocket-powered Vertical Takeoff and Vertical Landing (VTVL) vehicles for access to
suborbital and orbital space.

 The company's name refers to the blue planet, Earth, as the point of origin.
</item 6>

<item 7>
RAAF Base Amberley
Amberley was one of only two airfields in Australia (the other being Darwin International Airport) that were listed as a Transoceanic Abort (TOA) landing site
for the Space Shuttle.

 Amberley is currently undergoing a A$64 million dollar re-development program.

 The RAAF has plans to have Amberley operating as its "superbase" with flights of F/A-18F Super Hornets, F-35 Lightning II, KC-30A, C-17 Globemaster.

 No. 35 Squadron (Operating C-27J Spartan) is planned to move to the base from RAAF Base Richmond after it is fully equipped with the C-27J.
</item 7>

<item 8>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 8>

<item 9>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 9>

<item 10>
M7 Aerospace
M7 Aerospace LP is an aerospace company with its headquarters on the property of San Antonio International Airport in Uptown San Antonio, Texas, United States.
</item 10>
=== END ITEM id=143e2de2c16df8d9 ===

=== ITEM id=358d258d2ca624e1 ===
Gold sentences and facts:
Sentence 0: Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.
  Fact 0.0: Sleepless in Seattle is a 1993 film.
  Fact 0.1: Sleepless in Seattle is an American film.
  Fact 0.2: Sleepless in Seattle is a romantic comedy-drama film.
  Fact 0.3: Sleepless in Seattle was directed by Nora Ephron.
  Fact 0.4: Sleepless in Seattle was co-written by Nora Ephron.
  Fact 0.5: Sleepless in Seattle is based on a story by Jeff Arch.
Sentence 1:  It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber, and Rita Wilson.
  Fact 1.0: Sleepless in Seattle stars Tom Hanks.
  Fact 1.1: Sleepless in Seattle stars Meg Ryan.
  Fact 1.2: Sleepless in Seattle has a supporting cast featuring Bill Pullman.
  Fact 1.3: Sleepless in Seattle has a supporting cast featuring Ross Malinger.
  Fact 1.4: Sleepless in Seattle has a supporting cast featuring Rob Reiner.
  Fact 1.5: Sleepless in Seattle has a supporting cast featuring Rosie O'Donnell.
  Fact 1.6: Sleepless in Seattle has a supporting cast featuring Gaby Hoffmann.
  Fact 1.7: Sleepless in Seattle has a supporting cast featuring Victor Garber.
  Fact 1.8: Sleepless in Seattle has a supporting cast featuring Rita Wilson.
Sentence 2: Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.
  Fact 2.0: Roseann O'Donnell was born on March 21, 1962.
  Fact 2.1: Roseann O'Donnell is an American comedian.
  Fact 2.2: Roseann O'Donnell is an American actress.
  Fact 2.3: Roseann O'Donnell is an American author.
  Fact 2.4: Roseann O'Donnell is an American television personality.

Retrieved items:
<item 1>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 1>

<item 2>
Amy O'Neill
Amy O'Neill (born July 8, 1971) is an American performer and former actress.

 After appearing in several sitcoms and starring as Molly Stark on "The Young and the Restless" in 1986, she was cast in her notable role as Amy Szalinski in
the 1989 Disney film, "Honey, I Shrunk the Kids", for which she was nominated for a Young Artist Award.

 She reprised her role as Amy Szalinski in the 1992 sequel, "Honey, I Blew Up the Kid" and appeared as Lisa Barnes in "Where's Rodney?

".
</item 2>

<item 3>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 3>

<item 4>
Matthew Perry
Matthew Langford Perry (born August 19, 1969) is a Canadian-American actor, best known for his role as Chandler Bing on the long-running NBC television sitcom
"Friends", as well as his portrayal of Ron Clark in the 2006 television movie "The Ron Clark Story".
</item 4>

<item 5>
Kate O'Beirne
Kate Walsh O'Beirne (September 23, 1949 – April 23, 2017) was the President of National Review Institute.

 She was the Washington editor of "National Review".

 Her column, "Bread and Circuses," covered Congress, politics, and U.S. domestic policy.
</item 5>

<item 6>
Carrie Fisher
Carrie Frances Fisher (October 21, 1956 – December 27, 2016) was an American actress, writer, and humorist.

 Fisher was known for playing Princess Leia in the "Star Wars" films; a role for which she was nominated for three Saturn Awards.

 Her other film roles included "Shampoo" (1975), "The Blues Brothers" (1980), "Hannah and Her Sisters" (1986), "The 'Burbs" (1989), "When Harry Met Sally..."
(1989), "Soapdish" (1991) and "The Women" (2008 film).

 She was nominated twice for the Primetime Emmy Award for Outstanding Guest Actress in a Comedy Series for her performances on the television series "30 Rock"
and "Catastrophe".

 She was posthumously made a Disney Legend in 2017.
</item 6>

<item 7>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 7>

<item 8>
Rosie O'Donnell
Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.

 She has been a magazine editor and continues to be a celebrity blogger, a lesbian rights activist, a television producer, and a collaborative partner in the
LGBT family vacation company, R Family Vacations.
</item 8>

<item 9>
Barry Sandler
Barry Sandler (born February 23, 1947 in Buffalo, New York) is an American screenwriter and film producer.

 His career has spanned several decades, with the 1980s being his most prolific.

 The openly gay Sandler is perhaps best known for writing the 1982 film "Making Love", the first mainstream Hollywood film to deal seriously with issues of
homosexuality and coming out.

 Sandler discussed "Making Love" in the 1995 documentary film "The Celluloid Closet".
</item 9>

<item 10>
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 10>
=== END ITEM id=358d258d2ca624e1 ===

=== ITEM id=29b904997cfed985 ===
Gold sentences and facts:
Sentence 0:  The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
  Fact 0.0: The ground was home to the Boston Reds.
  Fact 0.1: The Boston Reds played in the Players' League in 1890.
  Fact 0.2: The Boston Reds played in the American Association in 1891.
Sentence 1:  It is home to the Port of Boston, a major shipping facility in the northeastern United States.
  Fact 1.0: The ground is home to the Port of Boston.
  Fact 1.1: The Port of Boston is a major shipping facility.
  Fact 1.2: The Port of Boston is in the northeastern United States.

Retrieved items:
<item 1>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 1>

<item 2>
Braves Field
Braves Field was a baseball park in the Northeastern United States, located in Boston, Massachusetts.

 Today the site is home to Nickerson Field on the campus of Boston University.

 The stadium was home of the Boston Braves of the National League from 1915–1952, prior to the Braves' move to Milwaukee in 1953.

 The stadium hosted the 1936 Major League Baseball All-Star Game and Braves home games during the 1948 World Series.

 The Boston Red Sox used Braves Field for their home games in the 1915 and 1916 World Series since the stadium had a larger seating capacity than Fenway Park.

 Braves Field was the site of Babe Ruth's final season, playing for the Braves in 1935.

 From 1929 to 1932, the Boston Red Sox played select regular season games periodically at Braves Field.

 On May 1, 1926, Braves Field hosted the longest baseball game in history – 26 Innings.

 The game ended in a 1–1 tie.
</item 2>

<item 3>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 3>

<item 4>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 4>

<item 5>
Deering Oaks
Deering Oaks is a 55 acre public park in Portland, Maine, which has a baseball diamond, tennis courts, a playground, and a pond.

 It is located west of downtown Portland and is bordered by Deering Avenue on the west, Forest Avenue on the east, Park Avenue to the south and Interstate 295
to the north.

 State Street bisects the park.

 Access is from State Street, Deering Avenue, or Park Avenue.

 The Portland Farmers Market is located on the Park Avenue side of the park.

 The park is listed on the National Register of Historic Places.
</item 5>

<item 6>
Transportation in Boston
The Boston transportation system includes roadway, subway, regional rail, air, and sea options for passenger and freight transit in Boston, Massachusetts.

 The Massachusetts Port Authority (Massport) operates the Port of Boston, which includes a container shipping facility in South Boston, and Logan International
Airport, in East Boston.

 The Massachusetts Bay Transportation Authority (MBTA) operates bus, subway, short distance rail, and water ferry passenger services throughout the city and
region.

 Amtrak operates passenger rail service to and from major northeastern cities.

 A major bus terminal at South Station is served by varied intercity bus companies.

 The city is bisected by major highways I-90 and I-93, the intersection of which has undergone a major renovation, nicknamed the Big Dig.
</item 6>

<item 7>
New Bedford Bay Sox
The New Bedford Bay Sox are a baseball team that plays in the New England Collegiate Baseball League, a collegiate summer baseball league located in the
northeastern United States region of New England.

 The team is located in New Bedford, Massachusetts.

 New Bedford's Paul Walsh Field serves as the home field of the Bay Sox.

 Th NECBL's players use wooden bats, as opposed to aluminum, and are college baseball players recruited from the top intercollegiate baseball programs in the
United States.

 The franchise has appeared in league finals four times in their history, all as the Torrington Twisters (1997, 1998, 2003, 2006), and have twice played host to
the NECBL All-Star Game as the Twisters (1998, 2008).

 Pat and Beth O'Connor, owners of Little Fenway and Little Wrigley joined the ownership team of the New Bedford Bay Sox in October, 2011, and signed a 3-year
lease to play at Paul Walsh Field.

 In November 2011, former MLB player Rick Miller was announced as the 2012 Bay Sox manager.
</item 7>

<item 8>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 8>

<item 9>
New Bedford Bay Sox
Miller was succeeded as manager prior to the 2015 season by Westport, MA native and former Boston Red Sox draft pick Kyle Fernandes.

 Fernandes has led the Bay Sox to back-to-back postseason appearances in his two seasons as manager, most recently losing in a three-game series to the eventual
NECBL Champions the Mystic Schooners.
</item 9>

<item 10>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 10>
=== END ITEM id=29b904997cfed985 ===

=== ITEM id=0d55e955014d80cd ===
Gold sentences and facts:
Sentence 0:  He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
  Fact 0.0: The person in this sentence succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.
Sentence 1: Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death in 1250.
  Fact 1.0: Eric IV is also known as Eric Ploughpenny.
  Fact 1.1: Eric IV is also known as Eric Plowpenny.
  Fact 1.2: Eric IV's Danish name is Erik Plovpenning.
  Fact 1.3: Eric IV's dates are 1216 to 10 August 1250.
  Fact 1.4: Eric IV was king of Denmark from 1241.
  Fact 1.5: Eric IV was king of Denmark until his death in 1250.

Retrieved items:
<item 1>
King Kang of Chu
King Kang of Chu (, died 545 BC) was from 559 to 545 BC the king of Chu, a major power during the Spring and Autumn period of ancient China.

 Born Xiong Zhao (), he succeeded his father King Gong of Chu.

 His reign was marked by constant wars with Jin, Chu's traditional enemy, and Wu, its new enemy.

 He ruled for 15 years and was succeeded by his son, Jia'ao, who would four years later be murdered by King Kang's younger brother King Ling of Chu.
</item 1>

<item 2>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 2>

<item 3>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 3>

<item 4>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 4>

<item 5>
Christopher I of Denmark
Christopher I (Danish: "Christoffer I" ) (1219 – 29 May 1259) was King of Denmark between 1252 and 1259.

 He was the son of Valdemar II of Denmark by his wife, Infanta Berengária of Portugal.

 He succeeded his brothers Eric IV Plovpenning and Abel of Denmark on the throne.

 Christopher was elected King upon the death of his older brother Abel in the summer of 1252.

 He was crowned at Lund Cathedral on Christmas Day 1252.
</item 5>

<item 6>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 6>

<item 7>
House of Oldenburg
The House of Oldenburg is a European dynasty of North German origin.

 It is one of Europe's most influential royal houses, with branches that rule or have ruled in Denmark, Iceland, Greece, Norway, Russia, Sweden, Schleswig,
Holstein, and Oldenburg.

 The current Queen of Denmark and King of Norway, the former King of Greece, the consorts of Greece and the United Kingdom, as well as the first twelve names in
the line of succession to the British throne, are all patrilineal members of the Glücksburg branch of this house.
</item 7>

<item 8>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 8>

<item 9>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 9>

<item 10>
Eric IV of Denmark
Eric IV, also known as Eric Ploughpenny or Eric Plowpenny (Danish: "Erik Plovpenning" ), (  1216 – 10 August 1250) was king of Denmark from 1241 until his death
in 1250.

 He was the son of King Valdemar II by his wife, Berengaria of Portugal, and brother of King Abel and King Christopher I.
</item 10>
=== END ITEM id=0d55e955014d80cd ===

=== ITEM id=d54ad9fd67951707 ===
Gold sentences and facts:
Sentence 0: Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.
  Fact 0.0: Hayley Nichole Williams was born on December 27, 1988.
  Fact 0.1: Hayley Nichole Williams is an American singer.
  Fact 0.2: Hayley Nichole Williams is an American songwriter.
  Fact 0.3: Hayley Nichole Williams is an American musician.
Sentence 1:  The band consists of Hayley Williams, Zac Farro and Taylor York.
  Fact 1.0: The band consists of Hayley Williams.
  Fact 1.1: The band consists of Zac Farro.
  Fact 1.2: The band consists of Taylor York.
Sentence 2: Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American heavy metal bands Lord Tracy and Pantera.
  Fact 2.0: Terrence Lee Glaze was born on November 29, 1964, in Columbus.
  Fact 2.1: Terrence Lee Glaze is known as Terry Lee.
  Fact 2.2: Terry Lee is a Texas-raised singer.
  Fact 2.3: Terry Lee is a Texas-raised guitarist.
  Fact 2.4: Terry Lee is most known for his work in the American heavy metal band Lord Tracy.
  Fact 2.5: Terry Lee is most known for his work in the American heavy metal band Pantera.

Retrieved items:
<item 1>
Richard Hawley
Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.

 After his first band Treebound Story (formed while he was still at school) broke up, Hawley found success as a member of Britpop band Longpigs in the 1990s.

 After that group broke up in 2000, he later joined the band Pulp, led by his friend Jarvis Cocker, for a short time.

 As a solo musician, Hawley has released seven studio albums.

 He has been nominated for a Mercury prize twice and once for a Brit Award.

 He has collaborated with Lisa Marie Presley, Arctic Monkeys, Manic Street Preachers and Paul Weller.
</item 1>

<item 2>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 2>

<item 3>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 3>

<item 4>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 4>

<item 5>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 5>

<item 6>
Hayley Williams (ice hockey)
Hayley Williams (born June 3, 1990) is an American professional ice hockey player for the Brampton Thunder of the Canadian Women's Hockey League (CWHL).

 Williams has previously played for the Buffalo Beauts in the National Women's Hockey League (NWHL).
</item 6>

<item 7>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 7>

<item 8>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 8>

<item 9>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 9>

<item 10>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 10>
=== END ITEM id=d54ad9fd67951707 ===
