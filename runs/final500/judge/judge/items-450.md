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

This file has 11 items.

=== ITEM id=8ada47e19ffacea6 ===
Gold sentences and facts:
Sentence 0: The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.
  Fact 0.0: The Livesey Hall War Memorial commemorates the fallen of World War I.
  Fact 0.1: The Livesey Hall War Memorial commemorates the fallen of World War II.
  Fact 0.2: The commemorated fallen had been employed by the South Suburban Gas Company of London.
Sentence 1:  Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).
  Fact 1.0: Over 60 million people were killed.
  Fact 1.1: The deaths were about 3% of the 1940 world population.
  Fact 1.2: The estimated 1940 world population was 2.3 billion.

Retrieved items:
<item 1>
Rainham War Memorial
The Rainham War Memorial commemorates soldiers killed in both World Wars as well as civilian casualties of World War II.
</item 1>

<item 2>
Arras Flying Services Memorial
The Arras Flying Services Memorial Commonwealth War Graves Commission war memorial in the Faubourg d'Amiens Cemetery, Arras, France.

 The memorial commemorates nearly 1,000 airmen from forces of the Commonwealth who were killed on the Western Front during World War I and who have no known
grave.

 The memorial was designed by Edwin Lutyens, sculpted by William Reid Dick and unveiled by Hugh Trenchard, 1st Viscount Trenchard, Marshal of the Royal Air
Force on 31 July 1932.
</item 2>

<item 3>
Portsmouth Naval Memorial
The Portsmouth Naval Memorial, sometimes known as Southsea Naval Memorial, is a war memorial in Portsmouth, Hampshire, England, on Southsea Common beside
Clarence Esplanade, between Clarence Pier and Southsea Castle.

 The memorial commemorates approximately 25,000 British and Commonwealth sailors who were lost in the World Wars, around 10,000 sailors in the First World War
and 15,000 in the Second World War.

 The memorial features a central obelisk, with names of the dead on bronze plaques arranged around the memorial according to the year of death.
</item 3>

<item 4>
Alamein Memorial
The Alamein Memorial is a Commonwealth War Graves Commission war memorial in the El Alamein War Cemetery, El Alamein, Egypt.

 The memorial commemorates 11,866 Commonwealth forces members who died during World War II.

 The memorial was designed by Hubert Worthington and unveiled by Viscount Montgomery of Alamein on 24 October 1954.
</item 4>

<item 5>
Tower Hill Memorial
The Tower Hill Memorial is a Commonwealth War Graves Commission war memorial on the south side of Trinity Square Gardens, in London, England.

 The memorial commemorates those from the Merchant Navy and fishing fleets who died during both world wars and have "no grave but the sea".

 The memorial was designed by Edwin Lutyens with sculpture work by William Reid Dick, the Second World War extension was designed by Edward Maufe with sculpture
work by Charles Wheeler.
</item 5>

<item 6>
Freetown Memorial
The Freetown Memorial is a Commonwealth War Graves Commission war memorial located outside the Secretariate Building in Freetown, Sierra Leone.

 The memorial generally commemorates Soldiers from Sierra Leone who died whilst serving with the Royal West African Frontier Force in West Africa and whose
graves are not known.

 The Freetown Memorial commemorates 1,109 First World War casualties and 247 from the Second World War.
</item 6>

<item 7>
War memorials honor military and civilian dead, including: Livesey Hall’s South Suburban Gas employees (both world wars; Sydney March); Alamein’s 11,866
Commonwealth dead (1954); Rainham’s soldiers and WWII civilians; Portsmouth’s roughly 25,000 sailors; Freetown’s 1,356 Sierra Leonean servicemen; Hanoi’s Second
Indochina War dead (1993); Tower Hill’s merchant seamen and fishers lost at sea; Arr
</item 7>

<item 8>
Livesey Hall War Memorial
The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.

 It is also a tribute to those employees who served in the wars.

 The monument was designed and executed by British sculptor Sydney March, of the March family of artists.
</item 8>

<item 9>
World War II caused an estimated 50–80+ million deaths: 50–55 million civilians and 21–25 million military personnel. Major Pacific battles included Bataan
(1942; mass surrender and Death March), Attu (1943), Saipan (1944), Manila (1945; over 100,000 civilians killed), Guam (1941 defeat, 1944 recapture), and
Okinawa (1945; 82 days). Other conflicts covered include
</item 9>

<item 10>
World War II casualties
World War II was the deadliest military conflict in history in absolute terms of total casualties.

 Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).

 The tables below give a detailed country-by-country count of human losses.

 World War II fatality statistics vary, with estimates of total deaths ranging from 50 million to more than 80 million.

 The higher figure of over 80 million includes deaths from war-related disease and famine.

 Civilians killed totalled 50 to 55 million, including 19 to 28 million from war-related disease and famine.

 Military deaths from all causes totalled 21 to 25 million, including deaths in captivity of about 5 million prisoners of war.
</item 10>
=== END ITEM id=8ada47e19ffacea6 ===

=== ITEM id=0bed60ca28fd18f3 ===
Gold sentences and facts:
Sentence 0: Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National Remembrance.
  Fact 0.0: Kolejka is a board game.
  Fact 0.1: Kolejka is Polish for 'queue' or 'line (of people)'.
  Fact 0.2: Karol Madaj designed Kolejka.
  Fact 0.3: Kolejka was launched in February 2011.
  Fact 0.4: Kolejka was launched by Poland's Institute of National Remembrance.
Sentence 1: Ghettopoly is a parody of "Monopoly" released in 2003.
  Fact 1.0: Ghettopoly is a parody of Monopoly.
  Fact 1.1: Ghettopoly was released in 2003.

Retrieved items:
<item 1>
Kick Over the Traces
Kick Over the Traces is a compilation album by the American punk rock band Pinhead Gunpowder.

 The album contains tracks from Pinhead Gunpowder releases since the band formed in 1990.

 The album was released through Recess Records on June 16, 2009.

 The Japanese edition features a second disc recorded live at 924 Gilman Street on February 10, 2008, and features different cover art.
</item 1>

<item 2>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 2>

<item 3>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 3>

<item 4>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 4>

<item 5>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 5>

<item 6>
Ghettopoly
Ghettopoly is a parody of "Monopoly" released in 2003.

 Invented by David Chang, it uses "Monopoly"-like mechanics in the atmosphere of a caricaturized United States ghetto.
</item 6>

<item 7>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 7>

<item 8>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 8>

<item 9>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 9>

<item 10>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 10>
=== END ITEM id=0bed60ca28fd18f3 ===

=== ITEM id=cd9078a40c10b664 ===
Gold sentences and facts:
Sentence 0:  It has an area of 1208 km2 and had a population of 600,000 in 2003.
  Fact 0.0: Ezhou has an area of 1208 km2.
  Fact 0.1: Ezhou had a population of 600,000 in 2003.
Sentence 1: Ezhou () is a prefecture-level city in eastern Hubei Province, China.
  Fact 1.0: Ezhou is a prefecture-level city.
  Fact 1.1: Ezhou is in eastern Hubei Province.
  Fact 1.2: Ezhou is in China.
Sentence 2:  As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.
  Fact 2.0: As of the 2010 census, Ezhou had a population of 1,048,668.
  Fact 2.1: As of the 2010 census, 668,727 of Ezhou's residents lived in the core Echeng District.
Sentence 3:  The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
  Fact 3.0: The Ezhou-Huanggang built-up (or metro) area was home to 1,035,496 inhabitants.
  Fact 3.1: The inhabitants of the Ezhou-Huanggang built-up area came from the Echeng and Huangzhou districts of Huanggang.

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
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 2>

<item 3>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 3>

<item 4>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 4>

<item 5>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 5>

<item 6>
Jiangyin dialect
Jiangyin dialect (江阴话) is a Northern Wu Chinese dialect spoken in the city of Jiangyin in Jiangsu province.

 Jiangyin dialect is a member of the Wu Chinese Taihu Wu family of dialects, which means the inhabitants speak a dialect similar to that of nearby Wuxi,
Changzhou, Suzhou, and Shanghai.

 Jiangyin dialect itself is of the Piling variety, related to the Changzhou dialect.

 Jiangyin dialect has the highest degree of mutual intelligibility with the dialects of the closest neighboring cities of Changzhou and Wuxi but also has a
fairly large degree of mutual intelligibility with the dialects of nearby Suzhou and Shanghai.

 As one travels south towards Wuxi away from the urban center of Jiangyin, Jiangyin dialect gradually becomes more and more closer sounding to the Wuxi dialect.
</item 6>

<item 7>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 7>

<item 8>
Ludhiana
Ludhiana is a city and a municipal corporation in Ludhiana district in the Indian state of Punjab, and is the largest city north of Delhi.

 It is the largest city in the state, with an area of 310sq.

 km and an estimated population of 1,789,650 as of the 2015.

 The population increases substantially during the harvesting season due to the migration of labourers from the eastern states of Uttar Pradesh, Bihar, Odisha.

 The city stands on the Sutlej River's old bank, 13 km south of its present course.

 It is an industrial centre of northern India, and was referred to as India's Manchester by the BBC.
</item 8>

<item 9>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 9>

<item 10>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 10>
=== END ITEM id=cd9078a40c10b664 ===

=== ITEM id=02fcd3c386c27034 ===
Gold sentences and facts:
Sentence 0:  They were first year members of the Mountain West Conference.
  Fact 0.0: They were first-year members of the Mountain West Conference.
Sentence 1: The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I Football Bowl Subdivision (FBS) (formerly I-A).
  Fact 1.0: The Mountain West Conference (MW) is a collegiate athletic conference.
  Fact 1.1: The Mountain West Conference is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA).
  Fact 1.2: The Mountain West Conference is affiliated with the NCAA Division I Football Bowl Subdivision (FBS).
  Fact 1.3: The Football Bowl Subdivision (FBS) was formerly called I-A.

Retrieved items:
<item 1>
2012 Hawaii Warriors football team
The 2012 Hawaii Warriors football team represented the University of Hawaii at Manoa in the 2012 NCAA Division I FBS football season.

 They were led by first-year head coach Norm Chow and played their home games at Aloha Stadium.

 They were first year members of the Mountain West Conference.

 They finished the season 3–9, 1–7 in Mountain West play to finish in a tie for ninth place.
</item 1>

<item 2>
Women's National Collegiate Athletic Association
The Women's National Collegiate Athletic Association (WNCAA) is an athletic association in the Philippines exclusively for women.

 It was founded in 1970.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 Its men's counterpart is the Men's National Collegiate Athletic Association, founded in 2004.
</item 2>

<item 3>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 3>

<item 4>
Wayne State Warriors football
The Wayne State Warriors football team is the college football team at Wayne State University in Detroit, Michigan.

 The Wayne State football team played their first game in October 1918.

 The Wayne State Warriors have competed in the Great Lakes Intercollegiate Athletic Conference since 1999 (and previously from 1975-1989), and are currently a
Division II member of the National Collegiate Athletic Association (NCAA).

 Wayne State plays their home games at Tom Adams Field at Wayne State Stadium.

 All Wayne State games are broadcast on WDTK radio.
</item 4>

<item 5>
List of Pacific Tigers head football coaches
The Pacific Tigers college football team represented University of the Pacific (CA).

 The Tigers competed in the National Collegiate Athletic Association (NCAA) College Division in the years 1937–1968.

 In 1969, the team moved to National Collegiate Athletic Association (NCAA) Division I and stayed there until disbanding after the 1995 season.
</item 5>

<item 6>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 6>

<item 7>
Hawaii Rainbow Warriors baseball
The Hawaii Rainbow Warriors baseball team is a varsity intercollegiate athletic team of the University of Hawaii at Manoa in Honolulu, Hawaii, United States.

 The team is a member of the Big West Conference, which is part of the National Collegiate Athletic Association's Division I.

 Hawaii's first baseball team was fielded in 1923.

 The team plays its home games at Les Murakami Stadium in Honolulu, Hawaii.

 The Rainbow Warriors are coached by Mike Trapasso.
</item 7>

<item 8>
Mountain West Conference
The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I
Football Bowl Subdivision (FBS) (formerly I-A).

 The MW officially began operations in July 1999.

 Geographically, the MW covers a broad expanse of the Western United States, with member schools located in California, Colorado, Hawaii, Idaho, Nevada, New
Mexico, Utah, and Wyoming.

 Craig Thompson has served as Commissioner of the MW since its founding in 1999.
</item 8>

<item 9>
Men's National Collegiate Athletic Association
The Men's National Collegiate Athletic Association is an athletic association in the Philippines, and is the direct counterpart of the older Women's National
Collegiate Athletic Association (founded in 1970).

 The Men's National Collegiate Athletic Association (MNCAA) is exclusive for men.

 The MNCAA was founded in 2004.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 The competing sports of the tournament are Basketball, Volleyball, Futsal, Table Tennis, Badminton and Streetdance.
</item 9>

<item 10>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 10>
=== END ITEM id=02fcd3c386c27034 ===

=== ITEM id=92e7f67377f661f7 ===
Gold sentences and facts:
Sentence 0: Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to 2003.
  Fact 0.0: Václav Havel was born on 5 October 1936.
  Fact 0.1: Václav Havel died on 18 December 2011.
  Fact 0.2: Václav Havel was a Czech writer.
  Fact 0.3: Václav Havel was a former dissident.
  Fact 0.4: Václav Havel served as the 1st President of the Czech Republic.
  Fact 0.5: Václav Havel served as President of the Czech Republic from 1993 to 2003.
Sentence 1: Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.
  Fact 1.0: Penelope Fitzgerald was born on 17 December 1916.
  Fact 1.1: Penelope Fitzgerald died on 28 April 2000.
  Fact 1.2: Penelope Fitzgerald was English.
  Fact 1.3: Penelope Fitzgerald was a Booker Prize-winning novelist.
  Fact 1.4: Penelope Fitzgerald was a poet.
  Fact 1.5: Penelope Fitzgerald was an essayist.
  Fact 1.6: Penelope Fitzgerald was a biographer.

Retrieved items:
<item 1>
Letters to Olga
Letters to Olga ("Czech:Dopisy Olze") is a book of compiled letters written by Czech playwright, dissident, and future president, Václav Havel to his wife Olga
Havlová during his nearly four-year imprisonment from May 1979 to March 1983.

 (Havel was released when he came down with a high fever and received a medical discharge).

 Havel was imprisoned by the communist regime of then Czechoslovakia for being one of the leaders of The Committee for the Defense of the Unjustly Prosecuted
(VONS) - most of whom had been signatories of the human rights document Charter 77.
</item 1>

<item 2>
Václav Havel Tribute Concert
Tribute to Václav Havel (Czech: Pocta Václavu Havlovi) was an event held in memory of Václav Havel, the last Czechoslovak and the first Czech President, writer,
playwright and human rights activist.

 The concert took place in Lucerna Music Bar in Prague on 23 December 2011.

 Václav Havel died on Sunday 18 December 2011 at 10:15 a.m.
</item 2>

<item 3>
Václav Havel Human Rights Prize
The Václav Havel Human Rights Prize is an annual €60,000 award established in 2013 by the Parliamentary Assembly of the Council of Europe, the Václav Havel
Library and the Charta 77 Foundation, to honour "outstanding" civil society action in defence of human rights, in Europe and beyond.
</item 3>

<item 4>
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
</item 4>

<item 5>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 5>

<item 6>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 6>

<item 7>
Ela, Hela and The Hitch
Ela, Hela, and the Hitch is a play by Václav Havel.

 The play was written for the Artistic Director of the Theatre on the Balustrade, Ivan Vyskočil, as part of a longer evening, entitled "Hitchhiking".

 Along with "Ela, Hela, and the Hitch", Havel also wrote a sketch called "Motormorphosis".

 Reportedly, Vyskočil altered Havel’s sketches for the performance, though the original text was discovered by a Czech theater scholar, Lenka Jungmannová.

 "Motormorphosis", in a translation by Carol Rocamora, was performed at the Havel Festival in 2006, a world premiere of the text as written.

 "Ela, Hela, and the Hitch" premiered in an English translation by Edward Einhorn following a revival of "Motormorphosis" at New York’s Bohemian National Hall
in 2011.
</item 7>

<item 8>
Vaněk plays
The Vaněk plays are a set of plays to which the character Ferdinand Vaněk is central.

 Vaněk first appeared in the play "Audience" by Václav Havel.

 He subsequently appeared in three other plays by Havel ("Protest", "Unveiling", and "Dozens of Cousins"), as well as plays by his friends and colleagues,
including Pavel Landovský and Tom Stoppard.
</item 8>

<item 9>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 9>

<item 10>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 10>
=== END ITEM id=92e7f67377f661f7 ===

=== ITEM id=152c259af90d45f7 ===
Gold sentences and facts:
Sentence 0: The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.
  Fact 0.0: The 1892 Auburn Tigers football team represented Auburn University.
  Fact 0.1: The 1892 Auburn Tigers football team played in the 1892 college football season.
Sentence 1:  It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.
  Fact 1.0: The 1892 Auburn Tigers football team was the first college football team fielded by the Agricultural and Mechanical College of Alabama.
  Fact 1.1: The Agricultural and Mechanical College of Alabama is now known as Auburn University.
Sentence 2:  With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in Alabama.
  Fact 2.0: Auburn University has more than 22,000 undergraduate students.
  Fact 2.1: Auburn University has a total enrollment of more than 28,000.
  Fact 2.2: Auburn University has 1,260 faculty members.
  Fact 2.3: Auburn University is the second largest university in Alabama.

Retrieved items:
<item 1>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 1>

<item 2>
1983 Alabama Crimson Tide football team
The 1983 Alabama Crimson Tide football team (variously "Alabama", "UA", "Bama" or "The Tide") represented the University of Alabama in the 1983 NCAA Division
I-A football season.

 It was the Crimson Tide's 91st overall and 50th season as a member of the Southeastern Conference (SEC).

 The team was led by head coach Ray Perkins, in his first year, and played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in
Birmingham, Alabama.

 They finished the season with a record of eight wins and four losses (8–4 overall, 4–2 in the SEC).

 Ray Perkins, who played as a wide receiver for Bear Bryant in the 1960s, was named as the new head coach at Alabama on December 14, 1982, to succeed Bryant
after his 25-year tenure as Alabama's head coach.
</item 2>

<item 3>
1892 Auburn Tigers football team
The 1892 Auburn Tigers football team represented Auburn University in the 1892 college football season.

 It was the first college football team fielded by the Agricultural and Mechanical College of Alabama, now known as Auburn University.

 The squad was first coached by George Petrie.

 Auburn shut out Georgia Tech, 26–0, just two days after being shut out themselves by North Carolina, 64–0.

 The team finished the season with a record of 2–2.
</item 3>

<item 4>
1902 Connecticut Aggies football team
The 1902 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1902 college football season.

 This was the seventh year that the school fielded a football team.

 The Aggies were led by first year head coach Edwin O. Smith, and completed the season with a record of 4–3.
</item 4>

<item 5>
1899 Connecticut Aggies football team
The 1899 Connecticut Aggies football team represented Connecticut Agricultural College, now the University of Connecticut, in the 1899 college football season.

 This was the fourth year that the school fielded a football team.

 The Aggies were led by first year head coach T. D. Knowles, and completed the season with a record of 6–2.
</item 5>

<item 6>
1905 Alabama Crimson White football team
The 1905 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1905 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 13th overall and 10th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach Jack Leavenworth, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of six wins and four losses (6–4 overall, 4–4 in the SIAA).
</item 6>

<item 7>
West Alabama Tigers football
The West Alabama Tigers football program is the intercollegiate American football team for the University of West Alabama located in the U.S. state of Alabama.

 The team competes in the NCAA Division II and are members of the Gulf South Conference.

 West Alabama's first football team was fielded in 1938.

 The team plays its home games at the 7,000 seat Tiger Stadium in Livingston, Alabama.

 The Tigers are coached by Brett Gilliland.
</item 7>

<item 8>
1906 Alabama Crimson White football team
The 1906 Alabama Crimson White football team (variously "Alabama", "UA" or "Bama") represented the University of Alabama in the 1906 Southern Intercollegiate
Athletic Association football season.

 It was Alabama's 14th overall and 11th season as a member of the Southern Intercollegiate Athletic Association (SIAA).

 The team was led by head coach J. W. H. Pollard, in his first year, and played their home games at both the University of Alabama Quad in Tuscaloosa and the
Birmingham Fairgrounds in Birmingham, Alabama.

 They finished the season with a record of five wins and one loss (5–1 overall, 3–1 in the SIAA).
</item 8>

<item 9>
Auburn University
Auburn University (AU or Auburn) is a public research university in Auburn, Alabama, United States.

 With more than 22,000 undergraduate students and a total enrollment of more than 28,000 with 1,260 faculty members, Auburn is the second largest university in
Alabama.

 Auburn University is one of the state's two public flagship universities.
</item 9>

<item 10>
1997 Alabama Crimson Tide football team
The 1997 Alabama Crimson Tide football team represented the University of Alabama in the college football season of 1997–98.

 The team's head coach was Mike DuBose, who was entering his first year at Alabama.

 They played their home games at both Bryant–Denny Stadium in Tuscaloosa and Legion Field in Birmingham, Alabama and competed in the West Division of the
Southeastern Conference.

 Alabama would finish with a record of 4–7 record in suffering the program's first losing season since the 1984 season.

 The loss against Kentucky marked Alabama's first ever overtime game, as overtime rules for college football had gone into effect the previous season.
</item 10>
=== END ITEM id=152c259af90d45f7 ===

=== ITEM id=a20228d778caad05 ===
Gold sentences and facts:
Sentence 0: William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story writer.
  Fact 0.0: William Somerset Maugham was born 25 January 1874.
  Fact 0.1: William Somerset Maugham died 16 December 1965.
  Fact 0.2: William Somerset Maugham is better known as W. Somerset Maugham.
  Fact 0.3: William Somerset Maugham was British.
  Fact 0.4: William Somerset Maugham was a playwright.
  Fact 0.5: William Somerset Maugham was a novelist.
  Fact 0.6: William Somerset Maugham was a short story writer.
  Fact 0.7: William Somerset Maugham's name includes the post-nominal CH.
Sentence 1:  He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
  Fact 1.0: W. Somerset Maugham was among the most popular writers of his era.
  Fact 1.1: W. Somerset Maugham was reputedly the highest-paid author during the 1930s.
Sentence 2: Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.
  Fact 2.0: Thomas Merton, O.C.S.O., was born January 31, 1915.
  Fact 2.1: Thomas Merton died December 10, 1968.
  Fact 2.2: Thomas Merton was an American Catholic writer.
  Fact 2.3: Thomas Merton was a theologian.
  Fact 2.4: Thomas Merton was a mystic.
Sentence 3:  In 1949, he was ordained to the priesthood and given the name Father Louis.
  Fact 3.0: In 1949, Thomas Merton was ordained to the priesthood.
  Fact 3.1: Thomas Merton was given the name Father Louis.

Retrieved items:
<item 1>
Trio (film)
Trio (also known as W. Somerset Maugham's Trio) is a 1950 British anthology film based on three short stories by W. Somerset Maugham: "The Verger", "Mr.
Know-All" and "Sanatorium".

 Ken Annakin directed "The Verger" and "Mr. Know-All", while Harold French was responsible for "Sanatorium".
</item 1>

<item 2>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 2>

<item 3>
Thomas Merton Award
The Thomas Merton Award has been awarded since 1972 by the Thomas Merton Center for Peace and Social Justice in Pittsburgh, United States.

 It is named after Thomas Merton and is given annually to "national and international individuals struggling for justice."
</item 3>

<item 4>
W. Somerset Maugham
William Somerset Maugham CH ( ; 25 January 1874 – 16 December 1965), better known as W. Somerset Maugham, was a British playwright, novelist and short story
writer.

 He was among the most popular writers of his era and reputedly the highest-paid author during the 1930s.
</item 4>

<item 5>
Somerset Maugham Award
The Somerset Maugham Award is a British literary prize given each May by the Society of Authors.

 It is awarded to the best writer or writers of a book published in the past year who is under the age of 35.

 The prize was instituted in 1947 by William Somerset Maugham and thus bears his name.

 The award is to be spent on foreign travel.

 The total fund for each year is £12,000.
</item 5>

<item 6>
Thomas Merton
Thomas Merton, O.C.S.O. (January 31, 1915 – December 10, 1968) was an American Catholic writer, theologian and mystic.

 A Trappist monk of the Abbey of Gethsemani, Kentucky, he was a poet, social activist, and student of comparative religion.

 In 1949, he was ordained to the priesthood and given the name Father Louis.
</item 6>

<item 7>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 7>

<item 8>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 8>

<item 9>
Michael Moorcock
Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.

 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
</item 9>

<item 10>
List of works about Thomas Merton
The following is a list of works about Thomas Merton, publications about Thomas Merton, the Trappist monk from Abbey of Gethsemani.

 The bibliography is organized into categories.

 A separate list of works by Thomas Merton is also available.
</item 10>
=== END ITEM id=a20228d778caad05 ===

=== ITEM id=7f9d1b104c0fa3cc ===
Gold sentences and facts:
Sentence 0:  He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).
  Fact 0.0: The player played three seasons for the Philadelphia Eagles.
  Fact 0.1: The player's Philadelphia Eagles seasons are listed as 1940.
  Fact 0.2: The player played for the Pittsburgh Steelers.
  Fact 0.3: The player's Pittsburgh Steelers seasons are listed as 1941-1942.
Sentence 1:  He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his departure from the Eagles.
  Fact 1.0: The player was drafted by the Philadelphia Eagles.
  Fact 1.1: The player was drafted in the second round of the 2008 NFL Draft.
  Fact 1.2: The player played for the Washington Redskins for three seasons.
  Fact 1.3: The player played for the Washington Redskins after leaving the Eagles.

Retrieved items:
<item 1>
Dallas Texans (NFL)
The Dallas Texans played in the National Football League (NFL) for one season, 1952, with a record of 1–11.

 The team is considered one of the worst teams in NFL history, both on (lowest franchise winning percentage) and off the field.

 The team was based first in Dallas, then Hershey, Pennsylvania, and Akron, Ohio, during its only season.

 The Texans were the last NFL team to fold.

 Many players on the 1952 roster went to the new Baltimore Colts franchise in 1953.

 The American Football League (AFL) had a 1960 charter member named the Dallas Texans (who later became the Kansas City Chiefs), but the AFL Texans have no
relationship with the earlier NFL team.
</item 1>

<item 2>
DeSean Jackson
DeSean William Jackson (born December 1, 1986) is an American football wide receiver for the Tampa Bay Buccaneers of the National Football League (NFL).

 He played college football for the University of California, Berkeley, where he was recognized as a consensus All-American.

 He was drafted by the Philadelphia Eagles in the second round of the 2008 NFL Draft, and played for the Washington Redskins for three seasons after his
departure from the Eagles.

 Jackson has been selected to the Pro Bowl three times, and was the first player selected to the Pro Bowl at two different positions in the same year when he
was named to the 2010 Pro Bowl as a wide receiver and return specialist.
</item 2>

<item 3>
Brooklyn Dodgers (NFL)
Later co-owner and partner Dan Topping (1912–1974), pulled the Tigers team out of the old NFL in 1946 and placed it in the newly established rival professional
league – the All-America Football Conference, which shortly lasted until 1949 until several stronger teams from the AAFC merged with and entered a reorganized
NFL in 1950.

 It lasted until 1970 with the NFL-AFL (third) merger following the establishment of the first "Super Bowl" inter-league national championship game three years
before with the old NFL champions playing the victors of the latest rival fourth incarnation of the American Football League IV, formed in 1960 (now the
American Football Conference (AFC).
</item 3>

<item 4>
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
</item 4>

<item 5>
Verlon Biggs
Verlon Marion Biggs (March 16, 1943 – June 7, 1994) was an American football defensive end in the American Football League and National Football League.

 He played for the New York Jets (AFL) in Super Bowl III, but felt he didn't receive enough credit for the Jets' playoff win against the Oakland Raiders in the
AFL Championship Game that launched them into the Super Bowl.

 He played well but sulked until 1970, demanded more money, and wound up signing with Vince Lombardi's Washington Redskins of the NFL.

 Always a dominating defensive end, Verlon solidified George Allen's defense (Allen replaced Lombardi upon his death in July 1970) and led the Redskins into
Super Bowl VII.

 His nickname with the Redskins was "dirty Biggs" because of his extremely physical style of play.
</item 5>

<item 6>
Jim O'Brien (American football)
Jim O'Brien (born February 2, 1947) is a former American football placekicker in the National Football League.

 He played for the Baltimore Colts from 1970 to 1972 and the Detroit Lions in 1973.

 He also played wide receiver, catching the bulk of his career passes during the 1972 season while still performing his kicking duties.

 His short career was less than stellar, posting a 55.6 percentage making 60 of 108 field goal attempts.

 His shining moment came in the closing moments of Super Bowl V in January 1971, where he kicked a 32-yard field goal with only five seconds remaining in the
game to break a tie and give the Colts the victory over the Dallas Cowboys 16–13.

 Before kicking the field goal, teammates saw that O'Brien was so nervous, he tried to take some of the artificial turf off the field to figure out the wind,
thinking the field was regular grass.

 Because of his singular moment kicking the Super Bowl-winning field goal, NFL Films named him the #9 "One-Hit Wonder" of all time.
</item 6>

<item 7>
Brooklyn Dodgers (NFL)
The Brooklyn Dodgers were an American football team that played in the National Football League from 1930 to 1943, and in 1944 as the Brooklyn Tigers.

 The team played its home games at Ebbets Field of the baseball National League's team, the Brooklyn Dodgers.

 In 1945, because of financial difficulties and the increasing scarcity of major league-level players because of the war-time defense requirements at the height
of World War II, the team was merged with the Boston Yanks and were known as the Yanks for that season.

 This old NFL franchise was not related to the earlier (second incarnation) American Football League II with a franchise that played as the Brooklyn Tigers for
the first half of the 1936 season before moving to Rochester, New York and playing as the Rochester Tigers.

 Another NFL team that played in the Brooklyn borough was the Brooklyn Lions (which became the Brooklyn Horsemen after merging with a team from an earlier first
incarnation AFL of the same name) in 1926.
</item 7>

<item 8>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 8>

<item 9>
1986 Tampa Bay Buccaneers season
The 1986 Tampa Bay Buccaneers season began with the team trying to improve on an 2–14 season.

 This was one of the worst seasons in franchise history.

 There is some sentiment that the 1986 team was even worse than the winless team of 1976, and the 473 points conceded was not beaten by any NFL team until the
2001 Indianapolis Colts gave up 486.

 The Buccaneers selected Bo Jackson with the top pick in the draft, but were unable to convince him to join the team.

 Three weeks after the draft, Jackson signed a three-year baseball contract with the Kansas City Royals.

 Despite holding four of the first forty selections in the draft, and the presence of a great influx of fresh talent from defunct USFL teams, the Buccaneers
were unable to find any impact players in either the draft or free agency.

 They entered the season with a roster nearly identical to the previous season's 2–14 team.
</item 9>

<item 10>
Don Looney
John Don Looney (September 2, 1916 – April 5, 2015) was a professional American football end in the National Football League.

 He was born in Sulphur Springs, Texas.

 He played three seasons for the Philadelphia Eagles (1940) and the Pittsburgh Steelers (1941–1942).

 He was the first receiver in NFL history to have over 100 yards receiving in each of his first two games, a feat which was not equaled until the 2008 NFL
season by another Eagles wide receiver, DeSean Jackson.

 At the time of his death, Looney was the second oldest living former NFL player.

 He was the father of NFL running back Joe Don Looney, who later died in a one-person motorcycle accident after his NFL career ended.
</item 10>
=== END ITEM id=7f9d1b104c0fa3cc ===

=== ITEM id=8adb3e17a80e39d3 ===
Gold sentences and facts:
Sentence 0: Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.
  Fact 0.0: The Cat in the Hat is a Dr. Seuss film.
  Fact 0.1: The Cat in the Hat is a 2003 film.
  Fact 0.2: The Cat in the Hat is an American film.
  Fact 0.3: The Cat in the Hat is a family comedy film.
  Fact 0.4: The Cat in the Hat was directed by Bo Welch.
Sentence 1: Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
  Fact 1.0: Robert W. 'Bo' Welch III was born November 30, 1951.
  Fact 1.1: Robert W. 'Bo' Welch III is an American production designer.
  Fact 1.2: Robert W. 'Bo' Welch III is an American director.

Retrieved items:
<item 1>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 1>

<item 2>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 2>

<item 3>
My Favorite Martian (film)
My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and
Ray Walston, based on the 1960s television series of the same name in which Walston starred.

 It was directed by Donald Petrie and written by Sherri Stoner and Deanna Oliver, based on the television series created by John L. Greene.

 Creatures were created by Amalgamated Dynamics from designs by Jordu Schell.

 The film grossed $36.8 million domestically against a budget of $65 million.
</item 3>

<item 4>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
</item 4>

<item 5>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 5>

<item 6>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 6>

<item 7>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 7>

<item 8>
The Cat in the Hat (film)
Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.

 It is based on the 1957 Dr. Seuss book "of the same name".

 The film stars Mike Myers in the title role of the Cat in the Hat, and Dakota Fanning as Sally.

 Sally's brother (who is unnamed in the book and the 1971 TV special), Conrad, is portrayed by Spencer Breslin.

 The film is the second feature-length Dr. Seuss adaptation after the 2000 holiday film "How the Grinch Stole Christmas".
</item 8>

<item 9>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 9>

<item 10>
Nitish Roy
Nitish Roy is an Indian film art director, production designer, and costume designer in Hindi cinema and a Bengali Film Director, who is also known for his work
with art cinema directors, Shyam Benegal, Mrinal Sen and Govind Nihlani, Hindi mainstream cinema, directors like Rajkumar Santoshi, as well as international
directors like Mira Nair and Gurinder Chadha.
</item 10>
=== END ITEM id=8adb3e17a80e39d3 ===

=== ITEM id=8b7439663d590422 ===
Gold sentences and facts:
Sentence 0:  All matches were played at HC Davos's home arena, Eisstadion Davos.
  Fact 0.0: All matches were played at the home arena of HC Davos.
  Fact 0.1: The home arena of HC Davos is Eisstadion Davos.
Sentence 1: Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.
  Fact 1.0: Hockey Club Davos is a professional Swiss ice hockey club.
  Fact 1.1: Hockey Club Davos is based in Davos, Switzerland.

Retrieved items:
<item 1>
HC Davos
Hockey Club Davos is a professional Swiss ice hockey club based in Davos, Switzerland.

 Davos plays in the National League (NL).

 The team is usually a strong force in the league and often bolster their roster with Swiss national team players and players who once played for National
Hockey League teams.

 HC Davos also has hosted the annual Spengler Cup tournament, an invitational competition, since 1923.
</item 1>

<item 2>
Spengler Cup
The Spengler Cup is an annual invitational ice hockey tournament held in Davos, Switzerland.

 First held in 1923, the Spengler Cup is often cited as the oldest invitational ice hockey tournament in the world.

 The event is hosted by the Swiss team HC Davos and played each year in Davos, Switzerland, between Christmas (December 25) and New Year's Day.

 Currently, all games are held at Vaillant Arena.
</item 2>

<item 3>
2015 Spengler Cup
The 89th Spengler Cup was an ice hockey competition held in Davos, Switzerland between 26 and 31 December 2015.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 3>

<item 4>
2016 Spengler Cup
The 90th Spengler Cup was an ice hockey competition being held in Davos, Switzerland between 26 and 31 December 2016.

 All matches were played at HC Davos's home known as Vaillant Arena.

 Six competing teams were split into two groups of three (in the round-robin series).

 The two groups, named Torriani and Cattini, are named after legendary Swiss hockey players Richard 'Bibi' Torriani and the Cattini brothers, Hans and
Ferdinand.
</item 4>

<item 5>
1992 Stanley Cup Finals
The 1992 Stanley Cup Finals NHL championship series was contested by the Prince of Wales Conference and defending Stanley Cup champion Pittsburgh Penguins and
the Clarence Campbell Conference champion Chicago Blackhawks.

 The Blackhawks were appearing in their first Finals since .

 After the Blackhawks jumped to an early 4–1 lead in the first game of the series, Mario Lemieux and the Penguins came back to win the game, sweep the series in
four games, and win their second consecutive and second overall Stanley Cup.

 It was the 99th year of the Stanley Cup, and the first to extend into the month of June.

 It was the last final for Chicago Stadium as it closed in 1994.
</item 5>

<item 6>
1992 Spengler Cup
The 1992 Spengler Cup was held in Davos, Switzerland, between December 26, 1992, and December 31, 1992.

 All matches were played at HC Davos's home arena, Eisstadion Davos.

 The final was won 6-5 by Team Canada over Färjestad BK.
</item 6>

<item 7>
Vaillant Arena
Vaillant Arena (Eisstadion Davos) is an arena in Davos, Switzerland.

 It is primarily used for ice hockey and is the home arena of HC Davos.

 It holds 7,080 people, of which 3,280 are seated.

 Every year the Spengler Cup is played in this arena.
</item 7>

<item 8>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 8>

<item 9>
Montreal Victorias
The Victoria Hockey Club of Montreal, Quebec, Canada was an early men's amateur ice hockey club.

 Its date of origin is ascribed to either 1874, 1877 or 1881, making it either the first or second organized ice hockey club after McGill University.

 The club played at its own rink, the Victoria Skating Rink in Montreal.

 The club was winners of the Stanley Cup in 1895 and held it except for a period in 1896 until 1899.

 The club remained amateur, splitting from the ranks of teams turned professional in 1908.

 The amateur hockey club was the first winner of the Allan Cup and continued in play until 1939 after its 65th season.

 The club often also fielded junior and intermediate teams.
</item 9>

<item 10>
1984 Stanley Cup Finals
This would be the second of eight consecutive Finals contested by a team from Alberta (the Oilers appeared in six, the Calgary Flames in two), and the first of
five consecutive Finals to end with the Cup presentation on Alberta ice (the Oilers won four times, the Montreal Canadiens one).
</item 10>
=== END ITEM id=8b7439663d590422 ===

=== ITEM id=87989e3782d497a1 ===
Gold sentences and facts:
Sentence 0:  He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth Games in Edinburgh, Scotland.
  Fact 0.0: He also participated in the 10,000 metres at the 1967 Pan American Games.
  Fact 0.1: The 1967 Pan American Games were held in Winnipeg, Manitoba, Canada.
  Fact 0.2: He represented Guyana at the 1970 British Commonwealth Games.
  Fact 0.3: The 1970 British Commonwealth Games were held in Edinburgh, Scotland.
Sentence 1:  The race consists of 25 laps around an Olympic-sized track.
  Fact 1.0: The race consists of 25 laps.
  Fact 1.1: The 25 laps are around an Olympic-sized track.

Retrieved items:
<item 1>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 1>

<item 2>
Dave Bailey (athlete)
David Bailey (born March 17, 1945 in Toronto, Ontario) is a retired track and field athlete, who represented Canada at the 1968 Summer Olympics in the men's
1.500 metres.

 He was the first Canadian to run the mile in less than 4 minutes (3:59.1) in San Diego, CA on June 11, 1966 and the first Canadian to the run the mile in less
than 4 minutes in Canada (3:57.7)in Toronto on July 22, 1967.

 A resident of Willowdale, Ontario he won the bronze medal in this event at the 1967 Pan American Games and the silver medal in this event at the 1968 World
University Games He was two-time Canadian Universities Track and Field Athlete of Year (1965, 1967) and two-time inductee into the University of Toronto Sports
Hall of Fame (individually 1998 and team 2003).
</item 2>

<item 3>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 3>

<item 4>
Nora Rocha
Nora Leticia Rocha de la Cruz (born December 18, 1967 in Monclova, Coahuila) is a retired female track and field athlete from Mexico, who competed in the 5000
and 10,000 metres.

 She claimed the gold medal in the women's 10,000 metres at the 1999 Pan American Games in Winnipeg, Canada.
</item 4>

<item 5>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 5>

<item 6>
Ross MacKenzie (athlete)
Ross MacKenzie (born 18 July 1946) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1968 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Brian MacLaren, Bill Crothers, and Robert McLaren).

 MacKenzie also finished fifth in the 1967 Pan American Games 400 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Brian MacLaren and Bill Crothers.
</item 6>

<item 7>
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 7>

<item 8>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 8>

<item 9>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 9>

<item 10>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 10>
=== END ITEM id=87989e3782d497a1 ===
