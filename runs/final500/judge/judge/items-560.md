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

This file has 10 items.

=== ITEM id=82155f7f26ba0f0a ===
Gold sentences and facts:
Sentence 0: Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).
  Fact 0.0: Groom Lake is a salt flat in Nevada.
  Fact 0.1: Groom Lake is used for runways of the Nellis Bombing Range Test Site airport.
  Fact 0.2: The Nellis Bombing Range Test Site airport has the code KXTA.
Sentence 1:  Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
  Fact 1.0: Groom Lake is located within the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.1: Groom Lake Valley is the namesake of the Groom Lake Valley portion of the Tonopah Basin.
  Fact 1.2: Groom Lake is 25 miles south of Rachel, Nevada.
Sentence 2: Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.
  Fact 2.0: Rachel is a census-designated place (CDP).
  Fact 2.1: Rachel is in Lincoln County, Nevada, United States.

Retrieved items:
<item 1>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 1>

<item 2>
Groom Lake (salt flat)
Groom Lake is a salt flat in Nevada used for runways of the Nellis Bombing Range Test Site airport (KXTA).

 It is located immediately north of the Area 51 USAF installation.

 The lake at 4409 ft elevation is approximately 3.7 mi from north to south and 3 mi from east to west at its widest point, and is approximately 11.3 miles in
circumference.

 Located within the namesake Groom Lake Valley portion of the Tonopah Basin, the lake is 25 mi south of Rachel, Nevada.
</item 2>

<item 3>
Papoose Lake
Papoose Lake is a dry lake bed located in Lincoln County, Nevada, USA.

 The lake lies within the plot of land referred to as the Groom Lake facility (aka Area 51) and is a restricted area.

 The lake is located a few miles southwest of the Groom lake facility nestled adjacent the Papoose Range.
</item 3>

<item 4>
Leviathan Mine
Leviathan Mine is a United States superfund site (CERCLIS ID: CAD98067685) at an abandoned open-pit sulfur mine located in Alpine County, California.

 The mine is located at on the eastern slope of the Sierra Nevada at about 7,000 ft elevation, 6 mi east of Markleeville and 24 mi southeast of Lake Tahoe.

 The mine site comprises approximately 250 acre of land surrounded by the Toiyabe National Forest, which is only accessible a few months a year.

 The approximately 22 million tons of sulfur ore-containing crushed rock at the mine are responsible for contaminating the Leviathan and Aspen Creek, which join
with Mountaineer Creek to form Bryant Creek which ultimately empties into the East Fork of the Carson River.

 These water bodies are listed as 303(d) impaired.

 The site location is seismically active.
</item 4>

<item 5>
John Skillern House
The John Skillern House is a historic cabin located 25 mi northwest of Fairfield in Camas County, Idaho, near the confluence of the Big Smokey and Little Smokey
creeks.

 The cabin was built in 1921-22 for John Skillern and his wife, who used it as a summer home and headquarters for John's large sheep ranching business.

 Skillern's wife based the cabin's rustic design off of the Old Faithful Inn in Yellowstone National Park, which is reflected in the cabin's steep roof and
covered front porch.

 The cabin's other significant rustic features include its sleeping lofts with pole railings, its horizontal log construction with exposed logs on the inner
walls, and its stone chimney.
</item 5>

<item 6>
Groom Range
The Groom Range is a mountain range in Lincoln County, Nevada.

 It is located within the Nevada Test and Training Range, north of Groom Dry Lake.

 The highest point in the Groom Range is 9,249 feet.

 The Groom Range is situated 26.6 miles north of the dry Groom Lake.
</item 6>

<item 7>
Rachel, Nevada
Rachel is a census-designated place (CDP) in Lincoln County, Nevada, United States.

 As of the 2010 census it had a population of 54.

 As the closest habitation to the Nellis Air Force Range and Area 51, Rachel enjoys a modest celebrity, particularly among aviation enthusiasts and UFO hunters.

 North of the town is the Quinn Canyon Range, which has the ghost town of Adaven.
</item 7>

<item 8>
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 8>

<item 9>
Saskatchewan Highway 35
Highway 35 is a paved undivided provincial highway in the Canadian province of Saskatchewan.

 It runs from the US Border near Port of Oungre (where it meets United States Route 85) to a dead end near the north shore of Tobin Lake.

 Saskatchewan Highway 35 is about 540 km long.

 The CanAm Highway comprises Saskatchewan Highways 35, SK Hwy 39, SK Hwy 6, SK Hwy 3, SK Hwy 2 and U.S. Route 85.

 46.5 mi of SK Hwy 35 contribute to the CanAm Highway between Port of Oungre on the Canada – United States border and Weyburn.

 Mudslides, and spring flooding were huge road building and maintenance problems around Nipawin as well as along the southern portion of the route named the
"Greater Yellow Grass Marsh".

 Over 20 early dams were built until the problem was addressed with the Rafferty-Alameda Project on the Souris River and the construction of the Qu'Appelle
River Dam which have helped to eliminate washed out roads and flooded communities.
</item 9>

<item 10>
Tonopah Test Range
Tonopah Test Range (TTR), is a restricted military installation located about 30 mi southeast of Tonopah, Nevada.

 It is part of the northern fringe of the Nellis Range, measuring 625 sqmi .

 Tonopah Test Range is located about 70 mi northwest of Groom Dry Lake, the home of the Area 51 facility.

 Like the Groom Lake facility, Tonopah is a site of interest to conspiracy theorists, mostly for its use of experimental and classified aircraft.

 As such, it is not the focus of alien enthusiasts, unlike its neighbor.

 It is currently used for nuclear weapons stockpile reliability testing, research and development of fusing and firing systems, and testing nuclear weapon
delivery systems.

 The airspace comprises restricted area R-4809 of the Nevada Test and Training Range and is often used for military training.
</item 10>
=== END ITEM id=82155f7f26ba0f0a ===

=== ITEM id=3fb22e05f015b107 ===
Gold sentences and facts:
Sentence 0:  The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".
  Fact 0.0: The album features a pair of duets with Waylon Jennings.
  Fact 0.1: One of the duets with Waylon Jennings was the "There Ain't No Good Chain Gang" single.
  Fact 0.2: The duet "There Ain't No Good Chain Gang" was one of Johnny Cash's first collaborations with Waylon Jennings.
  Fact 0.3: Johnny Cash and Waylon Jennings recorded songs together throughout the 1980s.
  Fact 0.4: Johnny Cash and Waylon Jennings recorded a separate album entitled "Heroes".
Sentence 1:  Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
  Fact 1.0: Johnny Cash and Waylon Jennings would also work together as The Highwaymen.
  Fact 1.1: The Highwaymen included Willie Nelson.
  Fact 1.2: The Highwaymen included Kris Kristofferson.
Sentence 2: The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
  Fact 2.0: The Highwaymen was a country music supergroup.
  Fact 2.1: The Highwaymen was composed of four of the country music genre's biggest artists.
  Fact 2.2: The four artists were well known for their pioneering influence on the outlaw country subgenre.
  Fact 2.3: The Highwaymen members were Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.
Sentence 3:  Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty Records.
  Fact 3.0: The Highwaymen was active as a group between 1985 and 1995.
  Fact 3.1: The Highwaymen recorded three major label albums.
  Fact 3.2: Two of The Highwaymen's major label albums were on Columbia Records.
  Fact 3.3: One of The Highwaymen's major label albums was for Liberty Records.

Retrieved items:
<item 1>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 1>

<item 2>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 2>

<item 3>
Confederate Railroad
Confederate Railroad is an American country rock–Southern rock band founded in 1987 in Marietta, Georgia, by Danny Shirley (lead vocals, rhythm guitar), Michael
Lamb (lead guitar), Mark Dufresne (drums), Chris McDaniel (keyboards), Warren "Gates" Nichols (steel guitar) and Wayne Secrest (bass guitar).

 After serving as a backing band for outlaw country acts David Allan Coe and Johnny Paycheck, the band signed to a recording contract with Atlantic Records,
releasing their self-titled debut album that year.

 In the 1990s, they released four more albums for Atlantic.
</item 3>

<item 4>
Back in the High Life Again
"Back in the High Life Again" is a 1986 single written by Steve Winwood and Will Jennings and performed by Winwood.

 The single was included on his album "Back in the High Life" and included backing vocals by James Taylor as well as a prominent mandolin played by Winwood.

 "Back in the High Life Again" was Winwood's second number one on the Adult Contemporary chart.

 The single stayed at number one for three weeks and went to number 13 on the "Billboard" Hot 100.
</item 4>

<item 5>
Leave Them Boys Alone
"Leave Them Boys Alone" is a song recorded by American country music artist Hank Williams Jr. with Waylon Jennings and Ernest Tubb.

 It was released in May 1983 as the second single from Williams' album "Strong Stuff".

 The song reached number 6 on the "Billboard" Hot Country Singles chart.

 It was written by Williams, Dean Dillon, Gary Stewart and Tanya Tucker.

 The song is notable for its combination of two singers associated with the outlaw movement with a country legend from the honky tonk days and golden age of the
Grand Ole Opry.

 Outlaw singers like Williams and Jennings saw themselves as taking country music back to its raw, honky tonk roots, and recording an up tempo song with Tubb
(who would never have received radio airplay in the late 1970s and early 80's) and reaching #6 was a slap in the face to the proponents of the country pop
sound.

 The lyrics of the song, much like Williams' "Family Tradition" echo the sentiment that the outlaw singers and their current escapades were predated by the hard
living honky tonkers of the 1950s such as Hank Williams, Sr.
</item 5>

<item 6>
I Would Like to See You Again
I Would Like to See You Again is the 57th album by American country singer Johnny Cash, released on Columbia Records in 1978.

 The title track peaked at #12 on the singles chart, while "There Ain't No Good Chain Gang" reached #2; the album itself peaked at #23.

 The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first
collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".

 Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
</item 6>

<item 7>
The Highwaymen (country supergroup)
The Highwaymen was a country music supergroup composed of four of the genre's biggest artists well known for their pioneering influence on the outlaw country
subgenre: Johnny Cash, Waylon Jennings, Willie Nelson, and Kris Kristofferson.

 Active as a group between 1985 and 1995, these four artists recorded three major label albums as The Highwaymen: two on Columbia Records and one for Liberty
Records.

 Their Columbia works produced three chart singles, including the number one "Highwayman" in 1985.
</item 7>

<item 8>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 8>

<item 9>
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 9>

<item 10>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 10>
=== END ITEM id=3fb22e05f015b107 ===

=== ITEM id=518974ab14537d8b ===
Gold sentences and facts:
Sentence 0: Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.
  Fact 0.0: Ranger Rick was originally called Ranger Rick's Nature Magazine
  Fact 0.1: Ranger Rick is a children's nature magazine
  Fact 0.2: Ranger Rick is published by the United States National Wildlife Federation
Sentence 1: TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.
  Fact 1.0: TENNIS is an American sports magazine
  Fact 1.1: TENNIS is owned by the Sinclair Broadcast Group

Retrieved items:
<item 1>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 1>

<item 2>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 2>

<item 3>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 3>

<item 4>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 4>

<item 5>
Graph query row:
entity: Ranger Rick
categories: ['__KGBuilder__', '__Entity__', 'CreativeWork']
kind: children's magazine
genre: None
relationships: ['PART_OF']
connected_entities: ['National Wildlife Federation']
</item 5>

<item 6>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 6>

<item 7>
Ranger Rick
Ranger Rick, originally Ranger Rick's Nature Magazine, is a children’s nature magazine that is published by the United States National Wildlife Federation.

 The magazine offers feature articles and activities for children, ages 7 and up, in order to spark their interest in the outdoors and become more actively
involved in the environment.

 The magazine's primary intention is to instill a passion for nature and promote activity outdoors.

 NWF also publishes two companion magazines, "Ranger Rick Jr.", which is aimed at ages 4–7, and "Ranger Rick Cub", which is aimed at kids 0–4 years old.)
</item 7>

<item 8>
Tennis (magazine)
TENNIS is an American sports magazine owned by the Sinclair Broadcast Group.

 There are eight issues per year covering the world of tennis.

 The "tennis.com" website is an offshoot of the magazine and part of the same media company.
</item 8>

<item 9>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 9>

<item 10>
Graph query row:
entity: TENNIS
categories: ['__KGBuilder__', '__Entity__', 'Thing']
kind: sports magazine
genre: None
relationships: ['PART_OF']
connected_entities: ['tennis.com']
</item 10>
=== END ITEM id=518974ab14537d8b ===

=== ITEM id=a67133e5efe2fc0a ===
Gold sentences and facts:
Sentence 0: Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
  Fact 0.0: Rolf Martin Theodor Dahlgren was born on 7 July 1932.
  Fact 0.1: Rolf Martin Theodor Dahlgren died on 14 February 1987.
  Fact 0.2: Rolf Martin Theodor Dahlgren was a Swedish-Danish botanist.
  Fact 0.3: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen.
  Fact 0.4: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen from 1973 until his death.
Sentence 1:  The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.
  Fact 1.0: The university has 23,473 undergraduate students.
  Fact 1.1: The university has 17,398 postgraduate students.
  Fact 1.2: The university has 2,968 doctoral students.
  Fact 1.3: The university has over 9,000 employees.

Retrieved items:
<item 1>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 1>

<item 2>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 2>

<item 3>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 3>

<item 4>
Rolf Dahlgren
Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
</item 4>

<item 5>
Ebba Witt-Brattström
Ebba Witt-Brattström (born 1953) is a Swedish scholar in comparative literature.

 She is Professor of Literature and head of department at Södertörn University outside Stockholm, and a well-known feminist.

 In the 1970s she was a member of the feminist organisation Grupp 8, and in 2005 she was one of the founders of the feminist political organisation and party
Feministiskt Initiativ, although she later distanced herself from the organisation and criticized its alleged strong left-wing tendencies.
</item 5>

<item 6>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 6>

<item 7>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 7>

<item 8>
William McGinnis
William "Bill" McGinnis, Ph.D. is a molecular biologist and professor of biology at the University of California San Diego.

 At UC San Diego he has also served as the Chairman of the Department of Biology from July 1998 - June 1999, as Associate Dean of the Division of Natural
Sciences from July 1, 1999 - June 2000, and as Interim Dean of the newly established Division of Biological Sciences from July 1, 2000 - February 1, 2001.

 Dr. McGinnis was appointed Dean of the Divisional Biological Sciences on July 1, 2013
</item 8>

<item 9>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 9>

<item 10>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 10>
=== END ITEM id=a67133e5efe2fc0a ===

=== ITEM id=6b3d3c92fe100cd1 ===
Gold sentences and facts:
Sentence 0:  The trains were built between 1994 and 2009 – for the most part by Talbot, part of Bombardier Inc., with some railcars built by De Dietrich.
  Fact 0.0: The trains were built between 1994 and 2009.
  Fact 0.1: The trains were built for the most part by Talbot.
  Fact 0.2: Talbot is part of Bombardier Inc.
  Fact 0.3: Some railcars were built by De Dietrich.
Sentence 1: Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.
  Fact 1.0: Bombardier Inc. is a multinational aerospace and transportation company.
  Fact 1.1: Bombardier Inc. is based in Montreal, Quebec, Canada.

Retrieved items:
<item 1>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 1>

<item 2>
Daimler AG
Daimler AG (] ) is a German multinational automotive corporation.

 Daimler AG is headquartered in Stuttgart, Baden-Württemberg, Germany.

 As of 2014, Daimler owns or has shares in a number of car, bus, truck and motorcycle brands including Mercedes-Benz, Mercedes-AMG, Smart Automobile, Detroit
Diesel, Freightliner, Western Star, Thomas Built Buses, Setra, BharatBenz, Mitsubishi Fuso, MV Agusta as well as shares in Denza, KAMAZ, Beijing Automotive
Group.

 The luxury Maybach brand was terminated at the end of 2012, but revived in April 2015 as "Mercedes-Maybach" versions of the Mercedes-Benz S-Class and G-Class.

 In 2016 Daimler sold 3.0 million vehicles.

 By unit sales, Daimler is the thirteenth-largest car manufacturer and is the largest truck manufacturer in the world.

 In addition to automobiles, Daimler manufactures buses and provides financial services through its Daimler Financial Services arm.

 The company is a component of the Euro Stoxx 50 stock market index.
</item 2>

<item 3>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 3>

<item 4>
Marquette Transportation Company
The Marquette Transportation Company is a marine transportation company based in Paducah, Kentucky, United States.
</item 4>

<item 5>
Volvo Halifax Assembly
The Volvo Halifax Assembly Plant located in Halifax, Nova Scotia was opened on 11 June 1963 by Prince Bertil.

 It was the first assembly plant Volvo opened outside of Sweden and the first non-domestic auto plant in North America.

 Volvo decided to open to the plant to bypass hefty North American import tariffs on foreign goods and to capitalize on the newly signed Canadian/American Auto
Pact.

 The plant was operated by Volvo Canada Limited (now Volvo Cars of Canada Corporation) in Toronto, Ontario and bridged the gap between Volvo of North America
(Rockleigh, New Jersey), Volvo headquarters and the flagship Torslanda plant in Gothenburg.
</item 5>

<item 6>
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
</item 6>

<item 7>
Volvo Car Gent
Volvo Car Gent (until 2007 known as Volvo Cars Europe Industry N.V.) is an automobile manufacturer located in the port district of Ghent in western Belgium.

 In recent years, the Ghent plant has become the largest Volvo plant in terms of the number of cars produced.

 The company is a wholly owned subsidiary of the Swedish automaker, Volvo Personvagnar AB (Volvo Cars).

 Legally it is part of Volvo Car Belgium NV.

 Its activities comprise welding, painting and final assembly.

 The plant became the only car manufacturing plant in Flanders after Ford Genk closed at the end of 2014.
</item 7>

<item 8>
Toland, Pennsylvania
Toland is less than 1/4 of a mile long, and it is located less than half a mile from the Appalachian trail crossing.

 The closest town is Mount Holly Springs, Pennsylvania located (3 miles north of Toland).

 The village is roughly 10 miles south of Carlisle, Pennsylvania, 7 miles south east of Boiling Springs, Pennsylvania, and 15 miles South of Gettysburg,
Pennsylvania.

 The only business in Toland is the Cherokee Campground formerly known as the Tagg Run Campground.

 The campground restaurant is now closed.

 Toland Mission is a small non-denominational church that can hold up to 74 persons.

 It was originally built by the owner of Beetem Lumber Company in Carlisle for the families of the community of Toland.

 A one-room school house near the church originally served the community, but it was converted to a home when residents' children were transported to a township
school.

 Local people in Toland usually travel to the town of Mount Holly Springs for necessities such as gas, food, and toiletries.
</item 8>

<item 9>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 9>

<item 10>
Northern Transportation Company
Northern Transportation Company Limited (NTCL) is a marine transportation company in the Canadian and American Arctic owned by Norterra, a holding company that
is owned by the Inuvialuit of the Northwest Territories.

 While primarily a marine freight hauler on Canadian rivers, like the Mackenzie and the Hay and along the Arctic coast of Canada, they are also a petroleum
wholesaler.

 Its head office is now in the town of Hay River, Northwest Territories.

 Today, it primarily utilizes tugs and barges.
</item 10>
=== END ITEM id=6b3d3c92fe100cd1 ===

=== ITEM id=2b4b9ad74111c60c ===
Gold sentences and facts:
Sentence 0: Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
  Fact 0.0: Janet Williams is an American soprano.
  Fact 0.1: Janet Williams has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Theatre des Champs-Elysees, Opera de Lyon, Nice Opera, Theatre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre.
  Fact 0.2: Janet Williams has performed in concerts throughout Europe, North America, Canada, Israel and Japan.
  Fact 0.3: Janet Williams has performed with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, Rene Jacobs, Marek Janowski, Neeme Jarvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
Sentence 1: Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized solo pianist, chamber music performer, and conductor.
  Fact 1.0: Vladimir Davidovich Ashkenazy's Russian name is Vladimir Davidovich Ashkenazi.
  Fact 1.1: Vladimir Davidovich Ashkenazy was born on 6 July 1937.
  Fact 1.2: Vladimir Davidovich Ashkenazy is an internationally recognized solo pianist.
  Fact 1.3: Vladimir Davidovich Ashkenazy is an internationally recognized chamber music performer.
  Fact 1.4: Vladimir Davidovich Ashkenazy is an internationally recognized conductor.
Sentence 2:  His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
  Fact 2.0: Vladimir Davidovich Ashkenazy's recordings have earned him five Grammy awards.
  Fact 2.1: Vladimir Davidovich Ashkenazy has received Iceland's Order of the Falcon.

Retrieved items:
<item 1>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 1>

<item 2>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 2>

<item 3>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 3>

<item 4>
Chico Hamilton Trio
Chico Hamilton Trio is an album by drummer and bandleader Chico Hamilton recorded at sessions in 1953, 1954 and 1956 released on the Pacific Jazz label.

 The album features Hamilton's first recordings for Pacific Jazz from 1953 and 1954, six tracks originally released on a 10-inch album, along with an additional
four recordings from 1956.
</item 4>

<item 5>
Fred Hersch
Fred Hersch (born October 21, 1955) is an American jazz pianist and educator.

 He has performed solo and led his own groups, including the Pocket Orchestra consisting of piano, trumpet, voice, and percussion.

 He was the first person to play weeklong engagements as a solo pianist at the Village Vanguard in New York City.

 He has recorded more than 70 of his jazz compositions.

 Hersch has been nominated for several Grammy Awards, and, as of December 2014, had been on the Jazz Studies faculty of the New England Conservatory since 1980
(with breaks).
</item 5>

<item 6>
The Impeccable Mr. Wilson
The Impeccable Mr. Wilson is an album by American jazz pianist Teddy Wilson featuring performances recorded in 1956 for the Norgran label.
</item 6>

<item 7>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 7>

<item 8>
Robert Glasper
Robert Glasper (born April 6, 1978, in Houston, Texas) is an American pianist and record producer.

 He has been nominated for 6 Grammys, has won 3 Grammy Awards and is currently nominated for an Emmy Award.

 His 2012 album "Black Radio" won the Grammy Award for Best R&B Album at the 55th Grammy Awards.

 His 2014 album "Black Radio 2" won the Grammy Award for Best Traditional R&B Performance at the 56th Grammy Awards.

 The song "These Walls" from Kendrick Lamar's album "To Pimp A Butterfly" won Best Rap/Sung Collaboration at the 57th Grammy Awards, on which Glasper plays
keys.

 The soundtrack for the film "Miles Ahead" won Best Soundtrack Compilation at the 58th Grammy Awards, for which Glasper was a producer.

 The song "Letter To The Free", written with Common, is nominated for an Emmy Award for Best Original Song in the Ava Duvernay documentary film "13th" (Netflix)
at the 2017 Emmys.
</item 8>

<item 9>
Janet Williams (soprano)
Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera,
Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco
Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with
conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard,
Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
</item 9>

<item 10>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 10>
=== END ITEM id=2b4b9ad74111c60c ===

=== ITEM id=5c87b9393b1ce074 ===
Gold sentences and facts:
Sentence 0:  The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover "Come Together".
  Fact 0.0: The tour accompanied the release of the piano-based album 'In the Running'.
  Fact 0.1: The tour featured pared-back versions of his hits.
  Fact 0.2: The tour featured pared-back versions of his album tracks.
  Fact 0.3: The tour featured a pared-back version of The Beatles cover 'Come Together'.
Sentence 1:  The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
  Fact 1.0: The song reached the top of the charts in the United States.
  Fact 1.1: The song peaked at No. 4 in the United Kingdom.

Retrieved items:
<item 1>
One Acoustic Night
One Acoustic Night is the third album live by the German hard rock band Bonfire.

 It was released in 2005 by E-M-S Music and is a two-disc album that is a live acoustic performance in Germany.

 It also features a studio song dedicated to the victims of the tsunami disaster in Thailand, recorded in English, German and Spanish.

 Chris Limburg and Thomas Streck were guest musicians that helped with the live performance, which is all in German.

 A DVD was released at the same time of the acoustic performance.
</item 1>

<item 2>
Beatles (song)
"Beatles", written by Sven Olov Bagge and Claes Bure, was the song that, performed by Swedish dansband Forbes, won the Swedish Melodifestivalen 1977.

 The song's lyrics are about the popular 1960s British rock group the Beatles and include the phrase "Yeah, yeah, yeah" as an homage to the Beatles song "She
Loves You".

 Forbes took the song to the United Kingdom for the Eurovision Song Contest 1977, with Anders Berglund as conductor.

 The audience cheered at the song, not unexpectedly since the contest took place in the United Kingdom, but the jury had another opinion.

 The song finished last, with only two points, from West Germany.
</item 2>

<item 3>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 3>

<item 4>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 4>

<item 5>
Back in the High Life Again
"Back in the High Life Again" is a 1986 single written by Steve Winwood and Will Jennings and performed by Winwood.

 The single was included on his album "Back in the High Life" and included backing vocals by James Taylor as well as a prominent mandolin played by Winwood.

 "Back in the High Life Again" was Winwood's second number one on the Adult Contemporary chart.

 The single stayed at number one for three weeks and went to number 13 on the "Billboard" Hot 100.
</item 5>

<item 6>
Baby, Stop Crying
"Baby, Stop Crying" is a song written by Bob Dylan, released in the summer of 1978 as a single and in a longer album version on "Street Legal".

 It was performed live only during his big-band worldwide tour of 1978.

 The song charted at #13 in the UK and was a top-ten song in much of Europe, although it failed to chart in the United States.
</item 6>

<item 7>
On Top of Old Smoky
"On Top of Old Smokey" is a traditional folk song of the United States.

 As recorded by The Weavers, the song reached the pop music charts in 1951.

 It is catalogued as Roud Folk Song Index No. 414.

 In one version the first verse is the following; for more on the words see below.
</item 7>

<item 8>
Range Life (song)
The single was not commercially released in the USA; it was issued by the band's UK label at the time, Big Cat.

 Both B-sides are outtakes from the "Crooked Rain, Crooked Rain" sessions and are included on the 2004 of that album.

 This song was one of many to be included in the group's greatest hits album "".
</item 8>

<item 9>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 9>

<item 10>
Live Acoustic America
Live Acoustic America is an acoustic live recording of classic Howard Jones songs, released in 1996.

 It was recorded in Los Angeles to a sell out crowd and features Carol Steele on percussion.

 The acoustic tour travelled the world, with Jones playing venues in Europe, the USA and Jamaica.

 The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover
"Come Together".
</item 10>
=== END ITEM id=5c87b9393b1ce074 ===

=== ITEM id=56b242d926fea655 ===
Gold sentences and facts:
Sentence 0: The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of its kind in Northern Europe.
  Fact 0.0: The Cathedral and Metropolitical Church of Saint Peter in York is commonly known as York Minster.
  Fact 0.1: York Minster is the cathedral of York, England.
  Fact 0.2: York Minster is one of the largest cathedrals of its kind in Northern Europe.
Sentence 1: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.
  Fact 1.0: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

Retrieved items:
<item 1>
York Minster astronomical clock
The York Minster astronomical clock was installed in the North Transept of York Minster in 1955.

 It was first conceived in 1944 and designed by R d'E Atkinson, chief assistant at the Royal Greenwich Observatory.

 The clock is a memorial to the airmen operating from bases in Yorkshire, Durham, and Northumberland who were killed in action during World War II.
</item 1>

<item 2>
York Minster
The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of
its kind in Northern Europe.

 The minster is the seat of the Archbishop of York, the second-highest office of the Church of England, and is the mother church for the Diocese of York and the
Province of York.

 It is run by a dean and chapter, under the Dean of York.

 The title "minster" is attributed to churches established in the Anglo-Saxon period as missionary teaching churches, and serves now as an honorific title.

 Services in the minster are sometimes regarded as on the High Church or Anglo-Catholic end of the Anglican continuum.
</item 2>

<item 3>
Lincoln Cathedral
Lincoln Cathedral or the Cathedral Church of the Blessed Virgin Mary of Lincoln, and sometimes St. Mary's Cathedral in Lincoln, England is the seat of the
Anglican bishop.

 Building commenced in 1088 and continued in several phases throughout the medieval period.

 It was the tallest building in the world for 238 years (1311–1549), and the first building to hold that title after the Great Pyramid of Giza.

 The central spire collapsed in 1549 and was not rebuilt.

 The cathedral is the third largest in Britain (in floor area) after St Paul's and York Minster, being 484 by .

 It is highly regarded by architectural scholars; the eminent Victorian writer John Ruskin declared: "I have always held... that the cathedral of Lincoln is out
and out the most precious piece of architecture in the British Isles and roughly speaking worth any two other cathedrals we have."
</item 3>

<item 4>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 4>

<item 5>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 5>

<item 6>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 6>

<item 7>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 7>

<item 8>
York
York ( ) is a historic walled city at the confluence of the rivers Ouse and Foss in North Yorkshire, England.

 The municipality is the traditional county town of the historic county of Yorkshire to which it gives its name.

 The city has a rich heritage and has provided the backdrop to major political events in England throughout much of its two millennia of existence.

 The city offers a wealth of historic attractions, of which York Minster is the most prominent, and a variety of cultural and sporting activities making it a
popular tourist destination for millions.
</item 8>

<item 9>
Minster (church)
Minster is an honorific title given to particular churches in England, most famously York Minster in York, Westminster in London and Southwell Minster in
Southwell.

 The term "minster" is first found in royal foundation charters of the 7th century.

 Although it corresponds to the Latin "monasterium" or monastery, it then designated any settlement of clergy living a communal life and endowed by charter with
the obligation of maintaining the daily office of prayer.

 Widespread in 10th-century Anglo-Saxon England, minsters declined in importance with the systematic introduction of parishes and parish churches from the 11th
century onwards.

 It continued as a title of dignity in later medieval England, for instances where a cathedral, monastery, collegiate church or parish church had originated
with an Anglo-Saxon foundation.

 Eventually a minster came to refer more generally to "any large or important church, especially a collegiate or cathedral church".

 In the 21st century, the Church of England has designated additional minsters by bestowing the status on existing parish churches.
</item 9>

<item 10>
Old Palace (York)
The Old Palace in the city of York, North Yorkshire, England, is also known as the Minster Library and is in Dean's Park.

 It houses York Minster’s library and archives as well as the Collections Department and conservation studio.

 Its name is a new one and renders homage to the part of the building that used to be the chapel of the Archbishop of York, which was built in the 13th century.

 It was refurbished in 1810 and shortly thereafter became the home of the Minster library.

 Notable items held in the collection include cathedral records dating to back to 1150 and a copy of the 1631 Wicked Bible.

 It is a Grade I listed building.

 An extension was added in 1998.
</item 10>
=== END ITEM id=56b242d926fea655 ===

=== ITEM id=505954807e931edc ===
Gold sentences and facts:
Sentence 0:  It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.
  Fact 0.0: The work is the third of Gluck's so-called reform operas for Vienna.
  Fact 0.1: The work follows 'Alceste' and 'Orfeo ed Euridice' among Gluck's reform operas for Vienna.
  Fact 0.2: The work is the least often performed of the three reform operas for Vienna.
Sentence 1: Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of Orpheus, set to a libretto by Ranieri de' Calzabigi.
  Fact 1.0: Orfeo ed Euridice is an opera composed by Christoph Willibald Gluck.
  Fact 1.1: Orfeo ed Euridice is based on the myth of Orpheus.
  Fact 1.2: Orfeo ed Euridice is set to a libretto by Ranieri de' Calzabigi.
  Fact 1.3: The French version of Orfeo ed Euridice is titled Orphée et Eurydice.
  Fact 1.4: The English title of Orfeo ed Euridice is 'Orpheus and Eurydice'.
Sentence 2:  The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.
  Fact 2.0: The piece was first performed at the Burgtheater in Vienna on 5 October 1762.
  Fact 2.1: The first performance of the piece was in the presence of Empress Maria Theresa.

Retrieved items:
<item 1>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 1>

<item 2>
Maria Theresa (film)
Maria Theresa (German: Maria Theresia) is a 1951 Austrian historical drama film directed by  Emil E. Reinert and starring  Paula Wessely, Fred Liewehr and
Marianne Schönauer.

 It portrays the life of the eighteenth century Habsburg Empress Maria Theresa.
</item 2>

<item 3>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 3>

<item 4>
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 4>

<item 5>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 5>

<item 6>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 6>

<item 7>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 7>

<item 8>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 8>

<item 9>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 9>

<item 10>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 10>
=== END ITEM id=505954807e931edc ===

=== ITEM id=0a67de5435cc88e2 ===
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
Transportation in Boston
The Boston transportation system includes roadway, subway, regional rail, air, and sea options for passenger and freight transit in Boston, Massachusetts.

 The Massachusetts Port Authority (Massport) operates the Port of Boston, which includes a container shipping facility in South Boston, and Logan International
Airport, in East Boston.

 The Massachusetts Bay Transportation Authority (MBTA) operates bus, subway, short distance rail, and water ferry passenger services throughout the city and
region.

 Amtrak operates passenger rail service to and from major northeastern cities.

 A major bus terminal at South Station is served by varied intercity bus companies.

 The city is bisected by major highways I-90 and I-93, the intersection of which has undergone a major renovation, nicknamed the Big Dig.
</item 1>

<item 2>
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
</item 2>

<item 3>
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
</item 3>

<item 4>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 4>

<item 5>
New Bedford Bay Sox
Miller was succeeded as manager prior to the 2015 season by Westport, MA native and former Boston Red Sox draft pick Kyle Fernandes.

 Fernandes has led the Bay Sox to back-to-back postseason appearances in his two seasons as manager, most recently losing in a three-game series to the eventual
NECBL Champions the Mystic Schooners.
</item 5>

<item 6>
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
</item 6>

<item 7>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 7>

<item 8>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 8>

<item 9>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 9>

<item 10>
The entries describe sports teams, venues, parks, and communities: Seattle Sounders FC 2, a Sounders reserve team founded in 2014; Charlotte Hornets; London
Skolars; Sacramento Republic FC, 2014 USL champions seeking MLS expansion; and several baseball, racing, and pipe-band organizations. Facilities include
Boston’s TD Garden, Braves Field site, and Camden Yards complex; BYU’s Smith Fieldhouse; Campeche’s Estadio Nelson Barrera; and Sacramento’s Papa
</item 10>
=== END ITEM id=0a67de5435cc88e2 ===
