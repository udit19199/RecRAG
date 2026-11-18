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

=== ITEM id=36788dac772e647a ===
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
Livesey Hall War Memorial
The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.

 It is also a tribute to those employees who served in the wars.

 The monument was designed and executed by British sculptor Sydney March, of the March family of artists.
</item 1>

<item 2>
Vietnam War Memorial, Hanoi
The War Memorial in Hanoi is located across the Ba Dinh Square, across the Ho Chi Minh Mausoleum and close to Hanoi Citadel.

 Constructed in 1993 in a fusion of traditional Vietnamese and modernist architecture, the memorial commemorates men and women who sacrificed themselves during
the Second Indochina War.

 The war is known by many names, e.g. as the American War in Vietnam.

 The memorial is a focal point for state functions, commemorating the war dead.
</item 2>

<item 3>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 3>

<item 4>
Rainham War Memorial
The Rainham War Memorial commemorates soldiers killed in both World Wars as well as civilian casualties of World War II.
</item 4>

<item 5>
Alamein Memorial
The Alamein Memorial is a Commonwealth War Graves Commission war memorial in the El Alamein War Cemetery, El Alamein, Egypt.

 The memorial commemorates 11,866 Commonwealth forces members who died during World War II.

 The memorial was designed by Hubert Worthington and unveiled by Viscount Montgomery of Alamein on 24 October 1954.
</item 5>

<item 6>
World War II casualties
World War II was the deadliest military conflict in history in absolute terms of total casualties.

 Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).

 The tables below give a detailed country-by-country count of human losses.

 World War II fatality statistics vary, with estimates of total deaths ranging from 50 million to more than 80 million.

 The higher figure of over 80 million includes deaths from war-related disease and famine.

 Civilians killed totalled 50 to 55 million, including 19 to 28 million from war-related disease and famine.

 Military deaths from all causes totalled 21 to 25 million, including deaths in captivity of about 5 million prisoners of war.
</item 6>

<item 7>
Tower Hill Memorial
The Tower Hill Memorial is a Commonwealth War Graves Commission war memorial on the south side of Trinity Square Gardens, in London, England.

 The memorial commemorates those from the Merchant Navy and fishing fleets who died during both world wars and have "no grave but the sea".

 The memorial was designed by Edwin Lutyens with sculpture work by William Reid Dick, the Second World War extension was designed by Edward Maufe with sculpture
work by Charles Wheeler.
</item 7>

<item 8>
Portsmouth Naval Memorial
The Portsmouth Naval Memorial, sometimes known as Southsea Naval Memorial, is a war memorial in Portsmouth, Hampshire, England, on Southsea Common beside
Clarence Esplanade, between Clarence Pier and Southsea Castle.

 The memorial commemorates approximately 25,000 British and Commonwealth sailors who were lost in the World Wars, around 10,000 sailors in the First World War
and 15,000 in the Second World War.

 The memorial features a central obelisk, with names of the dead on bronze plaques arranged around the memorial according to the year of death.
</item 8>

<item 9>
Freetown Memorial
The Freetown Memorial is a Commonwealth War Graves Commission war memorial located outside the Secretariate Building in Freetown, Sierra Leone.

 The memorial generally commemorates Soldiers from Sierra Leone who died whilst serving with the Royal West African Frontier Force in West Africa and whose
graves are not known.

 The Freetown Memorial commemorates 1,109 First World War casualties and 247 from the Second World War.
</item 9>

<item 10>
Arras Flying Services Memorial
The Arras Flying Services Memorial Commonwealth War Graves Commission war memorial in the Faubourg d'Amiens Cemetery, Arras, France.

 The memorial commemorates nearly 1,000 airmen from forces of the Commonwealth who were killed on the Western Front during World War I and who have no known
grave.

 The memorial was designed by Edwin Lutyens, sculpted by William Reid Dick and unveiled by Hugh Trenchard, 1st Viscount Trenchard, Marshal of the Royal Air
Force on 31 July 1932.
</item 10>
=== END ITEM id=36788dac772e647a ===

=== ITEM id=5a70aba117b0e893 ===
Gold sentences and facts:
Sentence 0: Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities (with notable exceptions).
  Fact 0.0: Kresy Wschodnie, also called Kresy, Eastern Borderlands, or Borderlands, was a region of the Second Polish Republic during the interwar period.
  Fact 0.1: Kresy constituted nearly half of the territory of the Second Polish Republic.
  Fact 0.2: In Kresy, ethnic Poles were the largest group.
  Fact 0.3: In Kresy, ethnic Poles were roughly equal in number to the size of the national minorities.
  Fact 0.4: The rough equality of ethnic Poles with the national minorities in Kresy had notable exceptions.
Sentence 1:  It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the Soviet Union.
  Fact 1.0: Kresy was drawn for the first time by the Supreme War Council.
  Fact 1.1: Kresy was drawn as the demarcation line between the newly emerging states, the Second Polish Republic and the Soviet Union.

Retrieved items:
<item 1>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 1>

<item 2>
List of Soviet and Eastern Bloc defectors
Soon after the formation of the Soviet Union, emigration restrictions were put in place to keep citizens from leaving the various countries of the Soviet
Socialist Republics, though some defections still occurred.

 During and after World War II, similar restrictions were put in place in non-Soviet countries of the Eastern Bloc, which consisted of the Communist states of
Eastern Europe.
</item 2>

<item 3>
Warsaw Pact
The Warsaw Pact, formally the Treaty of Friendship, Cooperation, and Mutual Assistance, was a collective defence treaty signed in Warsaw among the Soviet Union
and seven Soviet satellite states of Central and Eastern Europe during the Cold War.

 The Warsaw Pact was the military complement to the Council for Mutual Economic Assistance (CoMEcon), the regional economic organization for the socialist
states of Central and Eastern Europe.

 The Warsaw Pact was created in reaction to the integration of West Germany into NATO in 1955 per the London and Paris Conferences of 1954, but it is also
considered to have been motivated by Soviet desires to maintain control over military forces in Central and Eastern Europe.
</item 3>

<item 4>
Curzon Line
The history of the Curzon Line, with minor variations, goes back to the period following World War I.

 It was drawn for the first time by the Supreme War Council as the demarcation line between the newly emerging states, the Second Polish Republic, and the
Soviet Union.

 The proposal was put forward by British Foreign Secretary George Curzon, to serve as a diplomatic basis for the future border agreement, and in that form, it
never materialized because the war went on.
</item 4>

<item 5>
Kresy
In September 1939, after the Soviet Union joined Nazi Germany in their attack on Poland in accordance with the Molotov–Ribbentrop Pact, the territories were
incorporated into Soviet Ukraine, Belarus and Lithuania in the atmosphere of terror.
</item 5>

<item 6>
Kresy
Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly
half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities
(with notable exceptions).

 Administratively, the territory of Kresy was composed of voivodeships of Lwów, Nowogródek, Polesie, Stanisławów, Tarnopol, Wilno, Wołyń, and the Białystok.

 Today, these territories are divided between Western Ukraine, Western Belarus, and south-eastern Lithuania, with such major cities as Lviv, Vilnius, and Grodno
no longer in Poland.

 In the Second Polish Republic the term "Kresy" roughly equated with the lands beyond the so-called Curzon Line, which was suggested after World War I in
December 1919 by the British Foreign Office as the eastern border of the re-emerging sovereign Republic following the century of partitions.
</item 6>

<item 7>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 7>

<item 8>
Battle of Berestechko
The Battle of Berestechko (Polish: "Bitwa pod Beresteczkiem" ; Ukrainian: Берестецька битва, Битва під Берестечком ) was fought between the Ukrainian Cossacks,
led by Hetman Bohdan Khmelnytsky, aided by their Crimean Tatar allies, and a Polish army under King John II Casimir.

 It was a battle of a Cossack rebellion in Ukraine that took place in the years 1648–1657 after the expiration of a two-year truce.

 Fought from 28 to 30 June 1651, the battle took place in the province of Volhynia, on the hilly plain south of the Styr River.

 The Polish camp was on the river opposite Berestechko and faced south, towards the Cossack army about two kilometers away, whose right flank was against the
River Pliashivka (Pliashova) and the Tatar army on their left flank.

 It was probably the largest European land battle of the 17th century.
</item 8>

<item 9>
Eastern Bloc
The Eastern Bloc was the group of socialist states of Central and Eastern Europe, generally the Soviet Union and the countries of the Warsaw Pact.

 The terms Communist Bloc and Soviet Bloc were also used to denote groupings of states aligned with the Soviet Union, although these terms might include states
outside Central and Eastern Europe.
</item 9>

<item 10>
Flight and expulsion of Germans (1944–50)
During the later stages of World War II and the post-war period, German citizens and people of German ancestry were expelled from various Eastern European
countries and sent to the remaining territory of Germany and Austria.

 After 1950, some emigrated to the United States, Australia, and other countries from there.

 The areas affected included the former eastern territories of Germany, which were annexed by Poland and the Soviet Union after the war, as well as Germans who
were living within the prewar borders of Poland, Czechoslovakia, Hungary, Romania, Yugoslavia, and the Baltic States.

 The Nazis had made plans—only partially completed before the Nazi defeat—to remove many Slavic and Jewish people from Eastern Europe and settle the area with
Germans.

 The post war expulsion of the Germans formed a major part of the geopolitical and ethnic reconfiguration of Eastern Europe in the aftermath of World War II,
that attempted to create ethnically homogeneous nations within redefined borders.

 Between 1944 and 1948 about 31 million people, including ethnic Germans ('Volksdeutsche') as well as German citizens ('Reichsdeutsche'), were permanently or
temporarily moved from Central and Eastern Europe.
</item 10>
=== END ITEM id=5a70aba117b0e893 ===

=== ITEM id=993bb9e9b74cfe5e ===
Gold sentences and facts:
Sentence 0:  Levitt & Sons developed the Monmouth Heights section from 1965-1969.
  Fact 0.0: Levitt & Sons developed the Monmouth Heights section from 1965 to 1969.
Sentence 1:  The company is most famous for having built the town of Levittown, New York.
  Fact 1.0: Levitt & Sons is most famous for having built the town of Levittown, New York.

Retrieved items:
<item 1>
Monmouth Heights at Manalapan, New Jersey
Monmouth Heights at Manalapan is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 Levitt & Sons developed the Monmouth Heights section from 1965-1969.

 Residents of the 685 homes in the development are required to join the Monmouth Heights Community Association, which provides pools and other amenities.
</item 1>

<item 2>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 2>

<item 3>
Toland, Pennsylvania
Toland is an unincorporated community in Cumberland County, Pennsylvania, United States.

 The very small village is located on Pine Grove Road, east of Mountain Creek Campground.

 Michaux State Forest, the site of Laurel Lake, Fuller Lake and Pine Grove Furnace State Park lie a few miles to the west near the intersection of Pine Grove
Road and PA route 233.

 Toland has a mailing address of Gardners Pennsylvania, because the size of the community doesn't warrant a post-office, nor an official incorporated name.

 What is now a sand pit operation was originally a clay bank mining operation that was the reason for the location of the village of Toland.

 Less than 50 people reside in the close-knit community, which was built for the clay bank company workers in the first quarter of the 20th century.

 The original community of Toland, Pennsylvania consisted of 11 duplex houses, built side by side along Pine Grove Road, with less than 0.17 of an acre of land
to each.

 Additional homes have been built since.

 When the community was originally built there was only a common well with a hand pump for all of the families to draw from.
</item 3>

<item 4>
Levitt &amp; Sons
Levitt & Sons was a real estate development company founded by Abraham Levitt and later managed by his son William Levitt.

 The company is most famous for having built the town of Levittown, New York.

 The company's designs and building practices totally revolutionized the entire Home Building Industry and altered America's landscape with massive suburban
communities.
</item 4>

<item 5>
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 5>

<item 6>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 6>

<item 7>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 7>

<item 8>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 8>

<item 9>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 9>

<item 10>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 10>
=== END ITEM id=993bb9e9b74cfe5e ===

=== ITEM id=ddcb979e0adf5885 ===
Gold sentences and facts:
Sentence 0: Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the west.
  Fact 0.0: Longkou was formerly known as Huang County.
  Fact 0.1: Longkou is a port city.
  Fact 0.2: Longkou is in northeastern Shandong province, China.
  Fact 0.3: Longkou faces the Bohai Sea to the north.
  Fact 0.4: Longkou faces the Laizhou Bay to the west.
Sentence 1:  The district faces East China Sea and is connected to Mawei district by the Min River.
  Fact 1.0: The district faces the East China Sea.
  Fact 1.1: The district is connected to Mawei district by the Min River.

Retrieved items:
<item 1>
Dafeng District
Dafeng () is a coastal district under the administration of Yancheng, Jiangsu province, China.

 Located on the Jiangsu North Plain with a coastline of 112 km , Dafeng was historically one of the largest salt-making areas in China and now is famed for its
well preserved eco-system and numerous national conservation parks.

 The district has the largest national nature reserve for a rare deer species, Père David's Deer or Milu (麋鹿 ) in Chinese.

 It borders the prefecture-level city of Taizhou to the southwest.
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
Changle District
(, Foochow Romanized: Diòng-lŏ̤h) is a district located in eastern Fujian province, China.

 Administered by Fuzhou, it occupies a land area of 648 km2 and a sea area of 1327 km2 .

 Changle was established in the sixth year of Emperor Wu-De (623 AD) during the Tang Dynasty, and it became a county-level city on February 18, 1994.

 The district faces East China Sea and is connected to Mawei district by the Min River.

 Due to an increase in businesses, the province is now one of the richest province in China.

 The city was upgraded into a district on August 2017 by a government proposal.
</item 3>

<item 4>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 4>

<item 5>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 5>

<item 6>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 6>

<item 7>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 7>

<item 8>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 8>

<item 9>
Xujiacao Changle Station
Xujiacao Changle Station () is a station on Line 1 of the Ningbo Rail Transit that started operations on 30May 2014.

 It is situated over Wangchun Road () and Xinyuan Road () in the Haishu District of Ningbo City, Zhejiang Province, eastern China.
</item 9>

<item 10>
Line 6, Fuzhou Metro
Line 6 of the Fuzhou Metro () or Airport Line () runs from Cangshan District to Changle City.

 It starts at Exhibition Center Station and ends at Fuzhou Changle International Airport Station.

 The total length is 40.9 km.

 Line 6's color is pink.

 The construction is planned to begin by the end of 2016 and ends in 2021, from Exhibition Center to Fuzhou Changle International Airport.
</item 10>
=== END ITEM id=ddcb979e0adf5885 ===

=== ITEM id=8ae4cbd68e914bf6 ===
Gold sentences and facts:
Sentence 0: Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.
  Fact 0.0: Nelson Algren was born March 28, 1909.
  Fact 0.1: Nelson Algren died May 9, 1981.
  Fact 0.2: Nelson Algren was an American writer.
Sentence 1: Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
  Fact 1.0: Nathanael West was born Nathan Weinstein.
  Fact 1.1: Nathanael West was born October 17, 1903.
  Fact 1.2: Nathanael West died December 22, 1940.
  Fact 1.3: Nathanael West was an American author.
  Fact 1.4: Nathanael West was an American screenwriter.
  Fact 1.5: Nathanael West was an American satirist.

Retrieved items:
<item 1>
A Walk on the Wild Side
A Walk on the Wild Side is a 1956 novel by Nelson Algren, most often quoted as the source for Algren's "three rules of life": "Never play cards with a man
called Doc.

 Never eat at a place called Mom's.

 Never sleep with a woman whose troubles are worse than your own."
</item 1>

<item 2>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 2>

<item 3>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 3>

<item 4>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 4>

<item 5>
Fearless Frank
Fearless Frank is a 1967 film directed by Philip Kaufman.

 It is notable as the film debut of Jon Voight.

 Voight plays a murdered drifter who gets reanimated and turned into a superhero by a scientist (Severn Darden).

 Other notable cast members include "The Man With the Golden Arm" author Nelson Algren as a mobster named Needles, and "Word Jazz" vocal artist Ken Nordine as
the narrator, credited as "The Stranger."
</item 5>

<item 6>
The Man with the Golden Arm (novel)
The Man with the Golden Arm is a novel by Nelson Algren, published by Doubleday in November 1949.

 One of the seminal novels of post-World War II American letters, "The Man with the Golden Arm" is widely considered Algren's greatest and most enduring work.

 It won the National Book Award in 1950.
</item 6>

<item 7>
Hotel Kenmore Hall
Hotel Kenmore Hall is a 22-story single room occupancy hotel located at 145 East 23rd Street in the Gramercy section of Manhattan, designed by architect Maurice
Deutsch and constructed in 1927.

 Author Nathanael West lived and worked at the hotel as a night manager in the early years after the hotel opened; one of West's real-life experiences at the
hotel inspired the incident between Romola Martin and Homer Simpson that would later appear in "The Day of the Locust" (1939).

 West allowed friends like Edmund Wilson, Erskine Caldwell, S. J. Perelman and Maxwell Bodenheim free room and meals.

 Dashiell Hammett finished "The Maltese Falcon" here
</item 7>

<item 8>
Nelson Algren
Nelson Algren (March 28, 1909 – May 9, 1981) was an American writer.

 He may be best known for "The Man with the Golden Arm", a 1949 novel that won the National Book Award and was adapted as a 1955 film of the same name.
</item 8>

<item 9>
The Man with the Golden Arm
The Man with the Golden Arm is a 1955 American drama film with elements of film noir, based on the novel of the same name by Nelson Algren, which tells the
story of a drug addict who gets clean while in prison, but struggles to stay that way in the outside world.

 Although the addictive drug is never identified in the film, according to the American Film Institute "most contemporary and modern sources assume that it is
heroin", in contrast to Algren's book which named the drug as morphine.

 The film stars Frank Sinatra, Eleanor Parker, Kim Novak, Arnold Stang and Darren McGavin.

 It was adapted for the screen by Walter Newman, Lewis Meltzer and Ben Hecht (uncredited), and directed by Otto Preminger.

 The film's initial release sparked controversy due to its serious, in-depth treatment of the then-taboo subject of drug addiction.
</item 9>

<item 10>
Nathanael West
Nathanael West (born Nathan Weinstein; October 17, 1903 – December 22, 1940) was an American author, screenwriter and satirist.
</item 10>
=== END ITEM id=8ae4cbd68e914bf6 ===

=== ITEM id=4cb85f5f1b422cd9 ===
Gold sentences and facts:
Sentence 0: The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.
  Fact 0.0: The presidency of Ronald Reagan began on January 20, 1981
  Fact 0.1: Ronald Reagan was inaugurated as President of the United States on January 20, 1981
  Fact 0.2: The presidency of Ronald Reagan ended on January 20, 1989
Sentence 1: George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice President of the United States from 1981 to 1989.
  Fact 1.0: George Herbert Walker Bush was born June 12, 1924
  Fact 1.1: George Herbert Walker Bush is an American politician
  Fact 1.2: George Herbert Walker Bush was the 41st President of the United States from 1989 to 1993
  Fact 1.3: George Herbert Walker Bush was the 43rd Vice President of the United States from 1981 to 1989

Retrieved items:
<item 1>
Elliott Abrams
Elliott Abrams (born January 24, 1948) is a former American diplomat, lawyer and political scientist who served in foreign policy positions for Presidents
Ronald Reagan and George W. Bush.

 Abrams was convicted of withholding information from Congress about the Iran–Contra affair while serving under Reagan, but was pardoned by President George H.
W. Bush.
</item 1>

<item 2>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 2>

<item 3>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 3>

<item 4>
Reagan Democrat
A Reagan Democrat is a traditionally Democratic voter in the United States, referring especially to white working-class Rust Belt residents, who defected from
their party to support Republican President Ronald Reagan in either or both of the 1980 and 1984 elections as well as Republican Presidents George H. W. Bush in
the 1988 election and George W. Bush in either or both of the 2000 and 2004 elections.

 Part of this group also defected to Donald Trump, the Republican candidate in 2016, who won in Pennsylvania, Ohio, Michigan, Wisconsin, and Iowa, all states
that voted for Reagan but went Democratic in 2008 and 2012.
</item 4>

<item 5>
Ronald Reagan judicial appointment controversies
During President Ronald Reagan's presidency, he nominated at least twelve people for various federal appellate judgeship who were not confirmed.

 In some cases, the nominations were not processed by the Democratic-controlled Senate Judiciary Committee before Reagan's presidency ended, while in other
cases, nominees were rejected by the Senate Judiciary Committee or even blocked by unfriendly members of the Republican Party.

 Three of the nominees were renominated by Reagan's successor, President George H. W. Bush.

 Two of the nominees, Ferdinand Francis Fernandez and Guy G. Hurlbutt, were nominated after July 1, 1988, the traditional start date of the unofficial Thurmond
Rule during a presidential election year.

 Eight of the twelve seats eventually were filled by appointees of President George H. W. Bush.
</item 5>

<item 6>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 6>

<item 7>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 7>

<item 8>
First inauguration of Ronald Reagan
The first inauguration of Ronald Reagan as the 40th President of the United States was held on Tuesday, January 20, 1981, on the West Front of the United States
Capitol in Washington, D.C..

 This was the first inauguration to be held on the building's west side.

 The inauguration marked the commencement of the first four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.

 Chief Justice Warren E. Burger administered the presidential oath of office to Reagan, who placed his hand upon a family Bible given to him by his mother, open
to .

 Associate Justice Potter Stewart administered the vice presidential oath to Bush.
</item 8>

<item 9>
Presidency of Ronald Reagan
The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.

 Reagan, a Republican, took office as the 40th United States president following a landslide win over Democratic incumbent President Jimmy Carter in the 1980
presidential election.

 Reagan was succeeded by his vice president, George H. W. Bush, who won the 1988 presidential election with Reagan's support.
</item 9>

<item 10>
Second inauguration of Ronald Reagan
The second inauguration of Ronald Reagan as President of the United States was held privately on Sunday, January 20, 1985 in the Grand Foyer of the White House,
and publicly the following day, January 21, 1985 at the United States Capitol in Washington, D.C. The inauguration marked the commencement of the second
four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.
</item 10>
=== END ITEM id=4cb85f5f1b422cd9 ===

=== ITEM id=d660d47fe4bf17ff ===
Gold sentences and facts:
Sentence 0:  It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip Island.
  Fact 0.0: The area covers 1,383 sqkm.
  Fact 0.1: The area is south east of Melbourne.
  Fact 0.2: The area stretches from outer suburban Pakenham through rural areas.
  Fact 0.3: The area stretches to the coast at Inverloch and Phillip Island.
Sentence 1:  It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi, and part of suburban Pakenham.
  Fact 1.0: The area includes the town of Bass.
  Fact 1.1: The area includes the town of Cape Patterson.
  Fact 1.2: The area includes the town of Corinella.
  Fact 1.3: The area includes the town of Cowes.
  Fact 1.4: The area includes the town of Grantville.
  Fact 1.5: The area includes the town of Inverloch.
  Fact 1.6: The area includes the town of Kilcunda.
  Fact 1.7: The area includes the town of Koo Wee Rup.
  Fact 1.8: The area includes the town of Lang Lang.
  Fact 1.9: The area includes the town of Rhyll.
  Fact 1.10: The area includes the town of San Remo.
  Fact 1.11: The area includes the town of Ventnor.
  Fact 1.12: The area includes the town of Wonthaggi.
  Fact 1.13: The area includes part of suburban Pakenham.
Sentence 2: Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia local government area.
  Fact 2.0: Pakenham is a satellite suburb of Melbourne.
  Fact 2.1: Pakenham is on the edge of the West Gippsland region of Victoria, Australia.
  Fact 2.2: Pakenham is 56 km south-east of Melbourne's Central Business District.
  Fact 2.3: Pakenham is located within the Shire of Cardinia local government area.

Retrieved items:
<item 1>
Electoral district of Bass
The Electoral district of Bass is an electoral district of the Victorian Legislative Assembly.

 It covers an area of 1,383 sqkm south east of Melbourne, stretching from outer suburban Pakenham through rural areas to the coast at Inverloch and Phillip
Island.

 It includes the towns of Bass, Cape Patterson, Corinella, Cowes, Grantville, Inverloch, Kilcunda, Koo Wee Rup, Lang Lang, Rhyll, San Remo, Ventnor, Wonthaggi,
and part of suburban Pakenham.

 It lies within the Eastern Victoria Region of the upper house, the Legislative Council.
</item 1>

<item 2>
Electoral district of North Eastern Boroughs
North Eastern Boroughs was an electoral district of the Legislative Assembly in the Australian state of New South Wales from 1856 to 1859 that included the
towns of Newcastle, Stockton and Raymond Terrace.

 It was partly replaced by the electoral district of Newcastle and the electoral district of Hunter.
</item 2>

<item 3>
Electoral district of Emerald Hill
Emerald Hill was an electoral district of the Victorian Legislative Assembly, Australia.

 It covered part of the inner-city suburb South Melbourne and consisted part of the previous Electoral district of South Melbourne which was abolished in 1859.

 (The other part of the South Melbourne electorate became the Electoral district of Sandridge).
</item 3>

<item 4>
Electoral district of Braidwood
Braidwood was an electoral district of the Legislative Assembly in the Australian state of New South Wales between 1859 and 1904, which included the town of
Braidwood.

 It replaced parts of the electoral district of United Counties of Murray and St Vincent and the electoral district of Southern Boroughs.

 In 1904 it was largely absorbed into the electoral district of Queanbeyan.
</item 4>

<item 5>
Electoral district of Tullamarine
Electoral district of Tullamarine was an electoral district of the Legislative Assembly in the Australian state of Victoria.

 The district centred on the suburb of Tullamarine, about 17km north-west of Melbourne.
</item 5>

<item 6>
Pakenham, Victoria
Pakenham is a satellite suburb of Melbourne on the edge of the West Gippsland region of Victoria, Australia, 56 km south-east of Melbourne's Central Business
District, located within the Shire of Cardinia local government area.

 Pakenham recorded a population of 46,421 at the 2016 Census.
</item 6>

<item 7>
Officer, Victoria
Officer is a satellite suburb of Melbourne, Victoria, Australia, 54 km south-east of Melbourne's Central Business District, located within the Shire of Cardinia
local government area.

 Officer recorded a population of 7,133 at the 2016 Census.
</item 7>

<item 8>
Electoral district of Nunawading
The electoral district of Nunawading was an electoral district of the Legislative Assembly in the Australian state of Victoria, located in the eastern Melbourne
suburb of Nunawading.

 A notable former member was future Prime Minister, Robert Menzies.
</item 8>

<item 9>
Alpine National Park
The Alpine National Park is a national park located in the Central Highlands and Alpine regions of Victoria, Australia.

 The 646000 ha national park is located northeast of Melbourne.

 It is the largest National Park in Victoria, and covers much of the higher areas of the Great Dividing Range in Victoria, including Victoria's highest point,
Mount Bogong at 1986 m and the associated subalpine woodland and grassland of the Bogong High Plains.

 The park's north-eastern boundary is along the border with New South Wales, where it abuts the Kosciuszko National Park.

 On 7 November 2008 the Alpine National Park was added to the Australian National Heritage List as one of eleven areas constituting the Australian Alps National
Parks and Reserves.
</item 9>

<item 10>
Beaconsfield, Victoria
Beaconsfield is a satellite suburb of Melbourne, Victoria, Australia, 46 km south-east of Melbourne's Central Business District, located within the Shire of
Cardinia local government area.

 Beaconsfield recorded a population of 6,714 at the 2016 Census.
</item 10>
=== END ITEM id=d660d47fe4bf17ff ===

=== ITEM id=c1d028ca43748e3a ===
Gold sentences and facts:
Sentence 0: Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.
  Fact 0.0: Tritonia, also called flame freesia, is a genus of flowering plants.
  Fact 0.1: Tritonia belongs to the iris family.
  Fact 0.2: Tritonia was first described as a genus in 1802.
Sentence 1: Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
  Fact 1.0: Phyteuma is a genus of flowering plants.
  Fact 1.1: Phyteuma is in the family Campanulaceae.
  Fact 1.2: Phyteuma is native to Europe.
  Fact 1.3: Phyteuma is native to Morocco.

Retrieved items:
<item 1>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 1>

<item 2>
Thonningia
Thonningia is a monotypic genus of flowering plants in the family Balanophoraceae containing the single species Thonningia sanguinea.

 It is distributed throughout much of southern and western Africa, particularly the tropical regions.

 Common names for the plant include ground pineapple.

 A familiar plant to humans, it has an extremely long list of common names in many African languages.

 Many names are inspired by the resemblance of the plant's inflorescence to a pineapple or palm tree.

 Some of the names can be translated as pineapple of the bush (from Anyi), duiker's kolanut (from Igala), and crown of the ground (from Yoruba).
</item 2>

<item 3>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 3>

<item 4>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 4>

<item 5>
Vascular plant
Vascular plants (from Latin "vasculum": duct), also known as tracheophytes (from the equivalent Greek term "trachea") and also higher plants, form a large group
of plants (c. 308,312 accepted known species ) that are defined as those land plants that have lignified tissues (the xylem) for conducting water and minerals
throughout the plant.

 They also have a specialized non-lignified tissue (the phloem) to conduct products of photosynthesis.

 Vascular plants include the clubmosses, horsetails, ferns, gymnosperms (including conifers) and angiosperms (flowering plants).

 Scientific names for the group include Tracheophyta and Tracheobionta.
</item 5>

<item 6>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 6>

<item 7>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 7>

<item 8>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 8>

<item 9>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 9>

<item 10>
Phyteuma
Phyteuma is a genus of flowering plants in the family Campanulaceae, native to Europe and Morocco.
</item 10>
=== END ITEM id=c1d028ca43748e3a ===

=== ITEM id=4d1473af27e69892 ===
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
Oatka Creek
Oatka Creek ( ) is the third longest tributary of the Genesee River, located entirely in the Western New York region of the U.S. state of New York.

 From southern Wyoming County, it flows 58 mi to the Genesee near Scottsville, draining an area of 215 sqmi that includes all or part of 23 towns and villages
in Wyoming, Genesee, Livingston and Monroe counties as well.

 Its name means "leaving the highlands" or "approaching an opening" in Seneca.
</item 2>

<item 3>
New York State Canal System
The New York State Canal System (formerly known as the New York State Barge Canal) is a successor to the Erie Canal and other canals within New York.

 Currently, the 525 mi system is composed of the Erie Canal, the Oswego Canal, the Cayuga–Seneca Canal, and the Champlain Canal.

 In 2014 the system was listed as a national historic district on the National Register of Historic Places in its entirety, and in 2016 it was designated a
National Historic Landmark.
</item 3>

<item 4>
Tamiami Canal
The Tamiami Canal or C-4 Canal, is a canal located in southern Florida in the United States.

 It flows in a west to east direction from the western part of the state in the Everglades past the Miami International Airport to a salinity control center
near the Miami River.

 It averages 8 ft in depth and is over 100 ft wide in some areas.
</item 4>

<item 5>
Cayuga–Seneca Canal
The Cayuga–Seneca Canal is a canal in New York, United States.

 It is now part of the New York State Canal System.
</item 5>

<item 6>
Black Creek (Genesee River)
Black Creek is a tributary of the Genesee River that runs for 46 mi in western New York, United States.

 The creek begins in Middlebury in Wyoming County and runs north for roughly the first half of its course and east for the other half, eventually joining the
Genesee River in Chili, Monroe County.

 Its drainage area spans around 202 mi2 , which is largely rural and agricultural.

 As of 2000, 40,000 residents lived in the creek's drainage area.
</item 6>

<item 7>
Genesee Valley Park
Genesee Valley Park is located in the south side of Rochester, New York along the shores of the Genesee River.

 The New York State Barge Canal (the currently in-use portion of the Erie Canal) crosses the Genesee River within the park.

 The University of Rochester is located near the park's north entrance.
</item 7>

<item 8>
Beaver and Erie Canal
The Beaver and Erie Canal, also known as the Erie Extension Canal, was part of the Pennsylvania Canal system and consisted of three sections: the Beaver
Division, the Shenango Division, and the Conneaut Division.

 The canal ran 136 mi north–south near the western edge of the state from the Ohio River to Lake Erie through Beaver County, Lawrence County, Mercer County,
Crawford County, and Erie County, Pennsylvania.
</item 8>

<item 9>
Seneca River Crossing Canals Historic District
Seneca River Crossing Canals Historic District is a national historic district located at Montezuma and Tyre in Cayuga and Seneca Counties, New York.

 The district includes more than a mile of the Enlarged Erie Canal prism (built here between 1849 and 1857); towpath and heelpath; a drydock; the remains of the
Richmond (Montezuma) Aqueduct crossing the Seneca River; remnants of the original Erie Canal, built between 1817 and 1825 and including Lock #62 and piers of
the original mule bridge from that era; and a culvert that carries a stream beneath the Enlarged Erie Canal.
</item 9>

<item 10>
New York State Canal Corporation
The New York State Canal Corporation is responsible for the oversight, administration and maintenance of the New York State Canal System, which consists of the
Erie Canal, Cayuga–Seneca Canal, Oswego Canal and Champlain Canal.

 It is also involved with the development and maintenance of the New York State Canalway Trail and with the general development and promotion of the Erie Canal
Corridor as both a tourist attraction and a working waterway.
</item 10>
=== END ITEM id=4d1473af27e69892 ===

=== ITEM id=4d1ee477c0c80a49 ===
Gold sentences and facts:
Sentence 0:  He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
  Fact 0.0: The person referred to as 'He' has had a cameo acting role as a tailor in The Wolf of Wall Street.
  Fact 0.1: The person referred to as 'He' has had a cameo acting role as a tailor in The Good Shepherd.
Sentence 1: The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir of the same name by Jordan Belfort.
  Fact 1.0: The Wolf of Wall Street is a 2013 film.
  Fact 1.1: The Wolf of Wall Street is American.
  Fact 1.2: The Wolf of Wall Street is a biographical black comedy crime film.
  Fact 1.3: The Wolf of Wall Street was directed by Martin Scorsese.
  Fact 1.4: The Wolf of Wall Street was written by Terence Winter.
  Fact 1.5: The Wolf of Wall Street is based on the memoir of the same name by Jordan Belfort.

Retrieved items:
<item 1>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 1>

<item 2>
The Boys Who Cried Wolf
The Boys Who Cried Wolf (), also known as The Shepherd, is a 2015 South Korean crime thriller drama film.

 Written and directed by Kim Jin-hwang in his first feature-length for his Korean Academy of Film Arts (KAFA)'s final year undergraduate film project, it
depicts how a former stage actor becomes involved in a murder case.
</item 2>

<item 3>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 3>

<item 4>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 4>

<item 5>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 5>

<item 6>
A Walk in the Woods (film)
A Walk in the Woods is a 2015 American biographical comedy-drama film directed by Ken Kwapis, based on the 1998 book/memoir of the same name by Bill Bryson.

 The film stars Robert Redford, Nick Nolte and Emma Thompson.

 The film was released on September 2, 2015, by Broad Green Pictures.
</item 6>

<item 7>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 7>

<item 8>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 8>

<item 9>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 9>

<item 10>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 10>
=== END ITEM id=4d1ee477c0c80a49 ===

=== ITEM id=a0674ca8145020c1 ===
Gold sentences and facts:
Sentence 0: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 0.0: Henry Louis Mencken lived from September 12, 1880 to January 29, 1956.
  Fact 0.1: Henry Louis Mencken was an American journalist.
  Fact 0.2: Henry Louis Mencken was an American satirist.
  Fact 0.3: Henry Louis Mencken was an American cultural critic.
  Fact 0.4: Henry Louis Mencken was a scholar of American English.
Sentence 1: Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.
  Fact 1.0: Albert Camus lived from 7 November 1913 to 4 January 1960.
  Fact 1.1: Albert Camus was a French philosopher.
  Fact 1.2: Albert Camus was an author.
  Fact 1.3: Albert Camus was a journalist.

Retrieved items:
<item 1>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 1>

<item 2>
Reflections on the Guillotine
"Reflections on the Guillotine" is an extended essay written in 1957 by Albert Camus.

 In the essay Camus takes an uncompromising position for the abolition of the death penalty.

 Camus's view is similar to that of Cesare Beccaria and the Marquis de Sade, the latter having also argued that murder premeditated and carried out by the state
was the worst kind.

 Camus states that he does not base his argument on sympathy for the convicted but on logical grounds and on proven statistics.

 Camus also argues that capital punishment is an easy option for the government where remedy and reform may be possible.
</item 2>

<item 3>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 3>

<item 4>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 4>

<item 5>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 5>

<item 6>
The Stranger (novel)
L’Étranger (The Outsider [UK], or The Stranger [US]) is a 1942 novel by French author Albert Camus.

 Its theme and outlook are often cited as examples of Camus' philosophy of the absurd and existentialism, though Camus personally rejected the latter label.
</item 6>

<item 7>
The Guest
"The Guest" (French: "L'Hôte" ) is a short story by the French writer Albert Camus.

 It was first published in 1957 as part of a collection entitled "Exile and the Kingdom" ("L'exil et le royaume").

 The French title "L'Hôte" translates into both "the guest" and "the host" which ties back to the relationship between the main characters of the story.

 Camus employs this short tale to reflect upon issues raised by the political situation in French North Africa.

 In particular, he explores the problem of refusing to take sides in the colonial conflict in Algeria, something that mirrors Camus' own non-aligned stance
which he had set out in his Nobel Prize acceptance speech.
</item 7>

<item 8>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 8>

<item 9>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 9>

<item 10>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 10>
=== END ITEM id=a0674ca8145020c1 ===
