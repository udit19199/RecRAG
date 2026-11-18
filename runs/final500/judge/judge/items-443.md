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

=== ITEM id=449ae2b87a3d9415 ===
Gold sentences and facts:
Sentence 0: Fastbacks were a Seattle, Washington, punk rock band.
  Fact 0.0: Fastbacks were a punk rock band.
  Fact 0.1: Fastbacks were from Seattle, Washington.
Sentence 1: Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.
  Fact 1.0: Local Natives is an American indie rock band.
  Fact 1.1: Local Natives was previously known as Cavil at Rest.
  Fact 1.2: Local Natives is based in Silver Lake, Los Angeles, California, United States.

Retrieved items:
<item 1>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 1>

<item 2>
Fastbacks
Fastbacks were a Seattle, Washington, punk rock band.

 Formed in 1979 by songwriter/guitarist Kurt Bloch (born August 28, 1960), and friends Lulu Gargiulo (guitar and vocals, born October 12, 1960) and Kim Warnick
(bass and vocals, born April 7, 1959), they disbanded in 2001.

 Their sound mixed a generally punk rock approach to vocals and sound textures with poppy tunes and strong musicianship.
</item 2>

<item 3>
Hummingbird (Local Natives album)
Hummingbird is the second studio album by American indie rock band Local Natives, released on January 29, 2013 on Frenchkiss Records in the US, and on
Infectious Records in Europe.

 Produced by The National guitarist Aaron Dessner, the album was preceded by the singles, "Breakers" and "Heavy Feet."

 "Breakers" reached #12 on the Billboard Top 200.

 The song "Mt. Washington" was also featured in the 2015 video game "Life Is Strange".
</item 3>

<item 4>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 4>

<item 5>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 5>

<item 6>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 6>

<item 7>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 7>

<item 8>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 8>

<item 9>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 9>

<item 10>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 10>
=== END ITEM id=449ae2b87a3d9415 ===

=== ITEM id=8e857d5aa9029061 ===
Gold sentences and facts:
Sentence 0:  His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".
  Fact 0.0: The exploits of the man referred to here in that battle were described by Hal Moore.
  Fact 0.1: Hal Moore described the exploits in the book 'We Were Soldiers Once and Young'.
Sentence 1: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 1.0: Harold Gregory 'Hal' Moore, Jr. was born on February 13, 1922.
  Fact 1.1: Harold Gregory 'Hal' Moore, Jr. died on February 10, 2017.
  Fact 1.2: Harold Gregory 'Hal' Moore, Jr. was a United States Army lieutenant general.
  Fact 1.3: Harold Gregory 'Hal' Moore, Jr. was an author.

Retrieved items:
<item 1>
Nelson DeMille
Nelson Richard DeMille (born August 23, 1943) is an American author of action adventure and suspense novels.

 His novels include "Plum Island", "The Charm School", and "The Gold Coast".

 DeMille has also written under the pen names Jack Cannon, Kurt Ladner, Ellen Kay and Brad Matthews.
</item 1>

<item 2>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 2>

<item 3>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 3>

<item 4>
Myron F. Diduryk
Myron F. Diduryk (July 15, 1938 to April 24, 1970) was a Ukrainian-American United States Army Major, who played a key role as an infantry company commander in
the Battle of Ia Drang, the first major battle of the Vietnam War.

 His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".

 Moore said that Diduryk was, “… the finest battlefield company commander I had ever seen, bar none.”

 Diduryk was killed in action on his second tour in Vietnam.
</item 4>

<item 5>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 5>

<item 6>
Graph query row:
person: Hal Moore
military_rank: ['United States Army lieutenant general', 'author']
book: We Were Soldiers Once and Young
book_relationship: CREATED_BY
related_person: Myron F. Diduryk
related_person_book_relationship: None
person_relationship: None
</item 6>

<item 7>
William K. MacNulty
William Kirk MacNulty (May 22, 1892 - August 3, 1964) was a U. S. Marine with a long and distinguished record.

 He was a young U.S. Marine Corps Second Lieutenant during World War One and saw action at the Battle of the Argonne Forest.

 He served as a captain during the Second U.S. Nicaraguan Campaign (1926–33).

 During the Second World War as Lieutenant Colonel he commanded the U.S. Marine Corps defense of Guam against overwhelming Imperial Japanese forces during the
First Battle of Guam.

 He was incarcerated by the Japanese as a prisoners of war.

 He was promoted to Brigadier General during captivity and retired from military service in 1946.

 He is buried at the Golden Gate National Cemetery in San Bruno, San Mateo County, California.
</item 7>

<item 8>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 8>

<item 9>
Motor Boys
The Motor Boys were the heroes of a popular series of adventure books for boys at the turn of the 20th century issued by the Stratemeyer Syndicate under the
pseudonym of Clarence Young.

 This series was published by Cupples & Leon and was issued with dustjackets and glossy frontispiece.

 Howard Garis (author of the Uncle Wiggily stories) wrote many, if not all, of these stories.
</item 9>

<item 10>
Dispatches (book)
Dispatches is a New Journalism book by Michael Herr that describes the author's experiences in Vietnam as a war correspondent for "Esquire" magazine.

 First published in 1977, "Dispatches" was one of the first pieces of American literature that portrayed the experiences of soldiers in the Vietnam War for
American readers.
</item 10>
=== END ITEM id=8e857d5aa9029061 ===

=== ITEM id=2ea1ca84e4289ae4 ===
Gold sentences and facts:
Sentence 0:  Editors are based in Silver Spring, Maryland and Seoul, Korea.
  Fact 0.0: Editors are based in Silver Spring, Maryland.
  Fact 0.1: Editors are based in Seoul, Korea.
Sentence 1:  The magazine is headquartered in Baltimore, Maryland.
  Fact 1.0: The magazine is headquartered in Baltimore, Maryland.

Retrieved items:
<item 1>
Adventist HealthCare
Adventist HealthCare is a not-for-profit health services organization based in Gaithersburg, Maryland that employs more than 6,200 people and provides
healthcare for more than 400,000 individuals in the community each year.

 The primary service area for Adventist HealthCare is the Washington, D.C. metropolitan area.

 Despite similar names, it is not a part of the California-based Adventist Health, or the Florida-based Adventist Health System.
</item 1>

<item 2>
Seventh-day Adventist education
The Seventh-day Adventist educational system is part of the Seventh-day Adventist Church and overseen by the General Conference of Seventh-day Adventist located
in Silver Spring, Maryland.The educational system is the second-largest Christian school-system in the world, after the Roman Catholic system.
</item 2>

<item 3>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 3>

<item 4>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 4>

<item 5>
Ellendale, Delaware
Ellendale is a town in Sussex County, Delaware, United States.

 The population was 381 at the 2010 census, an increase of 16.5% since 2000.

 It is part of the Salisbury, Maryland-Delaware Metropolitan Statistical Area.

 Ellendale is the "Gateway to Delaware's Resort Beaches" because it is the town located on U.S. Highway 113, the resort area's westernmost border, and Delaware
Route 16, the resort area's northernmost border with the eastern border being the Delaware Bay and Atlantic Ocean and the southern border being the state line
with Maryland.

 Ellendale is home to the Philadelphia Bible College and the Harbor Christian Academy.
</item 5>

<item 6>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 6>

<item 7>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 7>

<item 8>
Douglass (Washington, D.C.)
Douglass is a residential neighborhood in Southeast Washington, D.C., on the eastern side of St. Elizabeths Hospital, on the border of the Congress Heights
Metro Station.

 It is bounded by Suitland Parkway to the north and east, Alabama Avenue to the south, and the St. Elizabeths campus to the west.
</item 8>

<item 9>
Adventist World
Adventist World is a monthly international magazine of the Seventh-day Adventist Church published by the Review and Herald Publishing Association.

 Editors are based in Silver Spring, Maryland and Seoul, Korea.
</item 9>

<item 10>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 10>
=== END ITEM id=2ea1ca84e4289ae4 ===

=== ITEM id=d3fae6b776b9000b ===
Gold sentences and facts:
Sentence 0:  His siblings were Abby, Nelson, Laurance, Winthrop, and David.
  Fact 0.0: Abby was his sibling.
  Fact 0.1: Nelson was his sibling.
  Fact 0.2: Laurance was his sibling.
  Fact 0.3: Winthrop was his sibling.
  Fact 0.4: David was his sibling.
Sentence 1: David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.
  Fact 1.0: David Rockefeller lived from June 12, 1915 to March 20, 2017.
  Fact 1.1: David Rockefeller was an American banker.
  Fact 1.2: David Rockefeller was chairman of Chase Manhattan Corporation.
  Fact 1.3: David Rockefeller was chief executive of Chase Manhattan Corporation.

Retrieved items:
<item 1>
David Rockefeller
David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.

 He was the oldest living member of the Rockefeller family and family patriarch from August 2004 until his death in March 2017.

 Rockefeller was a son of John D. Rockefeller Jr. and Abby Aldrich Rockefeller, and a grandson of John D. Rockefeller and Laura Spelman Rockefeller.
</item 1>

<item 2>
JPMorgan Chase
JPMorgan Chase & Co. is a U.S. multinational banking and financial services holding company headquartered in New York City.

 It is the largest bank in the United States, the world's sixth largest bank by total assets, with total assets of US$2.5 trillion, and the world's second most
valuable bank by market capitalization, after ICBC.

 It is a major provider of financial services, and according to "Forbes" magazine is the world's sixth largest public company based upon a composite ranking.

 The hedge fund unit of JPMorgan Chase is the second largest hedge fund in the United States.

 The company was formed in 2000, when Chase Manhattan Corporation merged with J.P. Morgan & Co.
</item 2>

<item 3>
Alida Rockefeller Messinger
Alida Ferry Rockefeller (born 1948) is an American philanthropist.

 She is the youngest daughter of John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992), and a fourth-generation member of the
Rockefeller family.

 Her brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).

 According to an account in "The New York Times", Alida was only five years old when her father began to teach her about philanthropy.

 She said:
</item 3>

<item 4>
Chase Bank
JPMorgan Chase Bank, N.A., doing business as Chase Bank, is a national bank that constitutes the consumer and commercial banking subsidiary of the U.S.
multinational banking and financial services holding company, JPMorgan Chase.

 The bank was known as Chase Manhattan Bank until it merged with J.P. Morgan & Co. in 2000.

 Chase Manhattan Bank was formed by the merger of the Chase National Bank and The Manhattan Company in 1955.

 The bank has been headquartered in Columbus, Ohio since its merger with Bank One Corporation in 2004.

 The bank acquired the deposits and most assets of Washington Mutual.
</item 4>

<item 5>
Rockefeller Brothers Fund
The Rockefeller Brothers Fund (RBF) is a philanthropic foundation created and run by members of the Rockefeller family.

 It was founded in New York City in 1940 as the primary philanthropic vehicle for the five third-generation Rockefeller brothers: John D. Rockefeller III,
Nelson, Laurance, Winthrop and David.

 It is distinct from the Rockefeller Foundation.

 The Rockefellers are an industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the late 19th and
early 20th centuries.
</item 5>

<item 6>
Walter V. Shipley
Walter V. Shipley (born November 2, 1935) was the former Chairman and Chief Executive Officer of Chase Manhattan Bank and, before that, the company with which
it merged Chemical Bank.

 Shipley was named chief executive of Chemical in 1981 and held the position through 1999 and remained at the bank as chairman through January 2000, just prior
to the bank's merger with J.P. Morgan & Co..

 During his 18-year tenure, Shipley oversaw Chemical's mergers with Texas Commerce Bank in 1987, Manufacturers Hanover in 1991 and Chase Manhattan Bank in 1996.
</item 6>

<item 7>
John D. Rockefeller III
John Davison Rockefeller III (March 21, 1906 – July 10, 1978) was a philanthropist and third-generation member of the prominent Rockefeller family.

 He was the eldest son of philanthropists John D. Rockefeller Jr. and Abby Aldrich Rockefeller.

 His siblings were Abby, Nelson, Laurance, Winthrop, and David.
</item 7>

<item 8>
Hope Aldrich Rockefeller
Hope Aldrich Rockefeller (born May 17, 1938) is an American newspaper publisher.

 She is the eldest daughter of philanthropist John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992).

 Her elder brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).
</item 8>

<item 9>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 9>

<item 10>
William B. Harrison Jr.
William B. Harrison Jr., born August 12, 1943, in Rocky Mount, North Carolina, is the former CEO and chairman of JPMorgan Chase.

 He attended high school at Virginia Episcopal School, where he was a basketball star.

 He attended the University of North Carolina at Chapel Hill, where he was admitted to the Zeta Psi fraternity.

 Having risen through the ranks of Chemical Bank before succeeding Walter V. Shipley during the Chemical Bank takeover of the Chase Manhattan Corporation, he
and Douglas A. Warner III, then CEO of J.P. Morgan & Co., were the principal architects of the US$30.9 billion Chase and J.P. Morgan & Co. merger of 2000.

 Harrison has been a director of the Firm or a predecessor institution since 1991.

 Harrison is also a director of Merck & Co., Inc.
</item 10>
=== END ITEM id=d3fae6b776b9000b ===

=== ITEM id=2c03f7562cd70fe9 ===
Gold sentences and facts:
Sentence 0:  He is the main vocalist of the boy group Highlight.
  Fact 0.0: The person described is the main vocalist of the boy group Highlight.
Sentence 1:  She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls' Generation, in 2007.
  Fact 1.0: The person described was a trainee at S.M. Entertainment's Starlight Academy during her middle school years.
  Fact 1.1: The person described debuted as a member of Girls' Generation, the girl group of S.M. Entertainment, in 2007.
  Fact 1.2: The person described debuted in Girls' Generation after her trainee years.

Retrieved items:
<item 1>
Kim Tae-yeon (actress)
Kim Tae-yeon (born January 3, 1976) is a South Korean actress.

 She began her entertainment career as a model, winning Model Line's 40th Fashion Model contest in 1996 and the Pantene Model contest sponsored by Ford Models
in 2000.

 Kim made her film debut in the highly controversial film "Lies" in 1999.
</item 1>

<item 2>
Kim Tae-yeon discography
Apart from soundtrack recordings, Taeyeon has also recorded duets with other artists, most notably "Like a Star" with The One and "Different" with Kim Bum-soo,
which peaked at numbers one and two on the Gaon Digital Chart, respectively.
</item 2>

<item 3>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 3>

<item 4>
Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo"), also known as simply SJ or SUJU, is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 4>

<item 5>
List of songs written by Super Junior
Super Junior (Korean: 슈퍼주니어 ; "Syupeo Junieo") is a South Korean boy band.

 Formed in 2005 by producer Lee Soo-man of S.M. Entertainment, the group comprised a total of thirteen members at its peak.

 Super Junior originally debuted with twelve members, consisting of leader Leeteuk, Heechul, Hangeng, Yesung, Kangin, Shindong, Sungmin, Eunhyuk, Siwon,
Donghae, Ryeowook and Kibum.

 Kyuhyun joined the group in 2006.
</item 5>

<item 6>
Kim Tae-yeon discography
Kim Tae-yeon, better known by the mononym Taeyeon, is a South Korean singer.

 Her discography consists of one studio album, two extended plays (EPs), twenty singles (including four as featured artist), and three promotional singles.

 She debuted as a member of South Korean girl group Girls' Generation in August 2007 and initially gained some popularity as a singer upon recording soundtrack
songs "If" for "Hong Gil Dong" and "Can You Hear Me" for "Beethoven Virus" (2008).

 She subsequently established herself as one of the most renowned vocalist on the South Korean music scene with further soundtrack recordings, notably "I Love
You" for "" (2010), "Missing You like Crazy" for "The King 2 Hearts", "Closer" for "To the Beautiful You" (2012), and "And One" for "That Winter, the Wind
Blows" (2013), all of which managed to enter the top ten of South Korea's Gaon Digital Chart.
</item 6>

<item 7>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 7>

<item 8>
The First Collage
The First Collage is the debut and first EP by Yang Yo-seob, released on November 26, 2012.
</item 8>

<item 9>
Yang Yo-seob discography
The discography of South Korean singer-songwriter, musical actor Yang Yo-seob consists of one extended play and two singles.
</item 9>

<item 10>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 10>
=== END ITEM id=2c03f7562cd70fe9 ===

=== ITEM id=44b6325b190ac9a7 ===
Gold sentences and facts:
Sentence 0:  The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
  Fact 0.0: The breed is used for hunting in packs.
  Fact 0.1: The breed descends from the old Hound of Saintonge type of large hunting dog.
Sentence 1: The Kritikos Lagonikos (Cretan Hound) (Greek:Kρητικός Λαγωνικός) is a hunting breed of dog from the island of Crete, in Greece.
  Fact 1.0: The Kritikos Lagonikos, also called the Cretan Hound, is a hunting breed of dog.
  Fact 1.1: The Kritikos Lagonikos has the Greek name Κρητικός Λαγωνικός.
  Fact 1.2: The Kritikos Lagonikos is from the island of Crete, in Greece.

Retrieved items:
<item 1>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 1>

<item 2>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 2>

<item 3>
Grand Anglo-Français Blanc et Noir
The Grand Anglo-Français Blanc et Noir (FCI No.323) translated into English as the Great Anglo-French White and Black Hound, is a breed of dog used in hunting
as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) Foxhounds.
</item 3>

<item 4>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 4>

<item 5>
Warren Hound
Warren Hound refers to the group of Mediterranean hare-hunting dogs.

 In Spanish, this kind of breeds are called "podencos" and in Portuguese, "podengos".

 Generally speaking, a warren hound has erect ears, a smooth (or in some cases, wire) coat and a whip-like tail.

 However, there are some exceptions: for instance, the Cretan Hound has usually rose-ears and a curly or curved tail, while the Barrocal Algarvio is
long-haired.

 Unlike hounds or sighthounds which rely on just one sense, warren hounds tend to work with three senses: smell, sight and hearing.
</item 5>

<item 6>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 6>

<item 7>
Grand Anglo-Français Blanc et Orange
The Grand Anglo-Français Blanc et Orange (FCI No.324) translated into English as the Great Anglo-French White and Orange Hound, is a breed of dog used in
hunting as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) foxhounds.
</item 7>

<item 8>
Chien Français Blanc et Orange
The Chien Français Blanc et Orange (FCI No.316) is a breed of dog of the scenthound type, originating in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 8>

<item 9>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 9>

<item 10>
Cretan Hound
The Kritikos Lagonikos (Cretan Hound) (Greek:Kρητικός Λαγωνικός) is a hunting breed of dog from the island of Crete, in Greece.

 It is considered
</item 10>
=== END ITEM id=44b6325b190ac9a7 ===

=== ITEM id=c5fa0b06d0cf1058 ===
Gold sentences and facts:
Sentence 0: The Segugio Italiano ] is an Italian breed of dog of the scenthound family.
  Fact 0.0: The Segugio Italiano is an Italian breed of dog.
  Fact 0.1: The Segugio Italiano belongs to the scenthound family.
Sentence 1: The Sapsali (삽살이) is a shaggy Korean breed of dog.
  Fact 1.0: The Sapsali is a shaggy Korean breed of dog.
  Fact 1.1: The Sapsali's Korean name is 삽살이.
Sentence 2:  Traditionally, these dogs were believed to dispel ghosts and evil spirits.
  Fact 2.0: Traditionally, Sapsali dogs were believed to dispel ghosts.
  Fact 2.1: Traditionally, Sapsali dogs were believed to dispel evil spirits.

Retrieved items:
<item 1>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 1>

<item 2>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 2>

<item 3>
Segugio Italiano
The Segugio Italiano ] is an Italian breed of dog of the scenthound family.

 It comes in both short-haired and wire-haired varieties.

 It is thought to be an ancient breed, descended in pre-Roman eras from progenitor scenthounds in ancient Egypt.

 In 2009 ENCI (the Italian Kennel Club) registered 4,500 specimens of the short-haired variety and 1,740 wire-haired specimens, making this one of the top ten
breeds in Italy.
</item 3>

<item 4>
Chiribaya Dog
The Chiribaya Dog (Spanish: "perro Chiribaya" ) or Peruvian shepherd dog ("perro pastor Peruano ") was a pre-Columbian breed of dog from the southwest of Peru,
identified by the 42 mummies discovered by anthropologist Sonia Guillén Oneglio in the Ilo District, Moquegua Region, on the south coast of Peru.

 It has been established that it was a llama herding dog.

 The dogs were not only an important part of the social structure of the ancient Peruvians, but they received special treatment after death as well.

 The dog variety has been referred to in various Spanish-language documentaries under different terms, such as "el perro pastor Chribaya" ('the Chiribaya
shepherd dog') and "pastor Peruano " ('Peruvian shepherd'), though the ancient Peruvians did not keep sheep.

 Its original name is unknown.
</item 4>

<item 5>
Sapsali
The Sapsali (삽살이) is a shaggy Korean breed of dog.

 The word is followed in Korean by either "gae" (meaning "dog") or the suffix "ee"/"i", but is most commonly romanized as "Sapsaree".

 Traditionally, these dogs were believed to dispel ghosts and evil spirits.
</item 5>

<item 6>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
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
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 8>

<item 9>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 9>

<item 10>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 10>
=== END ITEM id=c5fa0b06d0cf1058 ===

=== ITEM id=729e119e352a5145 ===
Gold sentences and facts:
Sentence 0: On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.
  Fact 0.0: On February 25, 2017, this person lost to Jarrett Hurd.
  Fact 0.1: The February 25, 2017 loss was for the International Boxing Federation light middleweight world title.
Sentence 1: Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
  Fact 1.0: Jarrett Hurd was born on August 31, 1990.
  Fact 1.1: Jarrett Hurd is an American light middleweight boxer.
  Fact 1.2: Jarrett Hurd is from Accokeek, Maryland.

Retrieved items:
<item 1>
Tony Harrison (boxer)
On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.

 The fight was originally supposed to be a title eliminator for the right to fight Jermall Charlo, but when Charlo vacated the title, the fight was elevated to
be for the vacant title.

 Harrison boxed well for majority of the first half of the fight but began to fade thereafter the finish came in round 9 when Hurd landed a right hand that
floored Harrison,
</item 1>

<item 2>
Nedal Hussein
Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki, and Oriental and Pacific Boxing Federation (OPBF) super featherweight
title against Takashi Uchiyama, his professional fighting weight varied from 117+1/4 lb , i.e. bantamweight to 132+3/4 lb , i.e. lightweight.
</item 2>

<item 3>
Richard Williams (boxer)
Richard "The Secret" Williams ( (1971--) 09 1971 (age (2017)-(1971)-((11)<(05)or(11)==(05)and(30)<(09)) ) is an English professional light middle/middle/super
middleweight boxer of the 1990s, 2000s and 2010s who won the International Boxing Organization (IBO) light middleweight title, and Commonwealth light
middleweight title (twice), and was a challenger for the International Boxing Organization (IBO) light middleweight title against Sergio Gabriel Martinez, and
British Boxing Board of Control (BBBofC) British middleweight title against Howard Eastman, his professional fighting weight varied from 149+3/4 lb , i.e. light
middleweight to 162+1/4 lb , i.e. super middleweight.

 Richard Williams is trained by Brian Lawrence, and managed by Barry Hearn, and John Rooney.
</item 3>

<item 4>
Jarrett Hurd
Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
</item 4>

<item 5>
Nedal Hussein
Nedal "Skinny" Hussein (born 1 December 1977) is an Australian professional bantam/super bantam/feather/super feather/Lightweight boxer of the 1990s and 2000s
who won the Australian super bantamweight title, Australian bantamweight title, International Boxing Federation (IBF) Pan Pacific featherweight title, World
Boxing Federation (WBF) featherweight title, World Boxing Union (WBU) super bantamweight title, World Boxing Organization (WBO) Asia Pacific super featherweight
title, International Boxing Federation (IBF) Pan Pacific super featherweight title, International Boxing Organization (IBO) Inter-Continental super
featherweight title, and Commonwealth super bantamweight title, and was a challenger for the World Boxing Council (WBC) International super bantamweight title
against Manny Pacquiao, World Boxing Council (WBC) super bantamweight title against Óscar Larios, World Boxing Organization (WBO) featherweight title against
Scott Harrison, Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki,
</item 5>

<item 6>
Scott Dixon (boxer)
"Super" Scott Dixon ( (1976--) 28 1976 (age (2017)-(1976)-((11)<(09)or(11)==(09)and(30)<(28)) ) ) born in Hamilton is a Scottish professional feather/super
feather/light/light welter/welter/light middle/middle/super middleweight boxer of the 1990s, 2000s and 2010s, who won the British Boxing Board of Control
(BBBofC) Scottish Area welterweight title, World Boxing Board (WBB) welterweight title, World Boxing Union (German Version) super middleweight title, and
Commonwealth welterweight title, and was a challenger for the World Athletic Association (WAA) welterweight title against Michael Carruth, BBBofC British
welterweight title against Derek Roche, World Boxing Federation (WBF) light middleweight title against Steve Roberts, World Boxing Organization (WBO)
Inter-Continental light middleweight title against Anthony Farnell, and World Boxing Union (WBU) light middleweight title against Mehrdad Takalobigashi, his
professional fighting weight varied from 125 lb , i.e.
</item 6>

<item 7>
Scott Dixon (boxer)
featherweight to 167+1/2 lb , i.e. Super middleweight.
</item 7>

<item 8>
Kevin Kelly (boxer)
Kevin "Bones" Kelly (born August 7, 1969 in La Perouse, New South Wales) is an Australian retired professional welter/light middle/middleweight boxer of the
1990s and 2000s who won the South Australia State welterweight title, New South Wales (Australia) State welterweight title, New South Wales (Australia) State
light middleweight title, Australian light middleweight title, Pan Asian Boxing Association (PABA) light middleweight title, World Boxing Union (WBU)
middleweight title, and Commonwealth light middleweight title (twice), and was a challenger for the World Boxing Association (WBA) World light middleweight
title against David Reid, and World Boxing Organization (WBO) Inter-Continental light middleweight title against Gary Lockett, his professional fighting weight
varied from 142 lb , i.e. welterweight to 158+1/4 lb , i.e. middleweight.
</item 8>

<item 9>
Andre Thysse
super middleweight to 175 lb , i.e. light heavyweight.
</item 9>

<item 10>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 10>
=== END ITEM id=729e119e352a5145 ===

=== ITEM id=594249288ae1fe98 ===
Gold sentences and facts:
Sentence 0: Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
  Fact 0.0: Miltoniopsis is a genus of orchids.
  Fact 0.1: Miltoniopsis is abbreviated Mltnps in horticultural trade.
  Fact 0.2: Miltoniopsis is native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
Sentence 1: Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few species extending north into temperate regions of Asia.
  Fact 1.0: Strobilanthes is a genus of about 350 species of flowering plants.
  Fact 1.1: Strobilanthes is in the family Acanthaceae.
  Fact 1.2: Strobilanthes is mostly native to tropical Asia and Madagascar.
  Fact 1.3: A few Strobilanthes species extend north into temperate regions of Asia.

Retrieved items:
<item 1>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 1>

<item 2>
Hemigraphis
Hemigraphis is a genus of plants in the family Acanthaceae, consisting of about 30 species native to tropical Asia.

 "Hemigraphis" is sometimes included in the genus "Strobilanthes"
</item 2>

<item 3>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 3>

<item 4>
Miltoniopsis
Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.

 This genus comprises 5 species.

 Miltoniopsis's common name is Pansy Orchid.
</item 4>

<item 5>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 5>

<item 6>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
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
Miltoniopsis vexillaria
Miltoniopsis vexillaria ("the flag-like "Miltoniopsis"") is a species of epiphytic orchid in the genus Miltoniopsis.
</item 8>

<item 9>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 9>

<item 10>
Strobilanthes japonica
Strobilanthes japonica is a flowering herbaceous perennial plant from Asia, one of around 350 plants of the genus "Strobilanthes".

 The 20–50 cm ornamental plant is cultivated in Japan and China, and blooms in autumn with 1.5 cm purple to white funnel-shaped flowers.
</item 10>
=== END ITEM id=594249288ae1fe98 ===

=== ITEM id=6aa8a197980c9b81 ===
Gold sentences and facts:
Sentence 0: Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.
  Fact 0.0: Willi Forst was born Wilhelm Anton Frohs.
  Fact 0.1: Willi Forst's dates are 7 April 1903 to 11 August 1980.
  Fact 0.2: Willi Forst was an Austrian actor.
  Fact 0.3: Willi Forst was a screenwriter.
  Fact 0.4: Willi Forst was a film director.
  Fact 0.5: Willi Forst was a film producer.
  Fact 0.6: Willi Forst was a singer.
Sentence 1: Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.
  Fact 1.0: Elmer Clifton's dates are March 14, 1890 to October 15, 1949.
  Fact 1.1: Elmer Clifton was an American writer.
  Fact 1.2: Elmer Clifton was an American director.
  Fact 1.3: Elmer Clifton was an American actor.
  Fact 1.4: Elmer Clifton was active from the early silent days.

Retrieved items:
<item 1>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 1>

<item 2>
Lois Weber
Lois Weber (June 13, 1879 – November 13, 1939) was an American silent film actress, screenwriter, producer, and director, who is considered "the most important
female director the American film industry has known", and "one of the most important and prolific film directors in the era of silent films".

 Film historian Anthony Slide asserts that: "Along with D.W. Griffith, Weber was the American cinema's first genuine auteur, a filmmaker involved in all aspects
of production and one who utilized the motion picture to put across her own ideas and philosophies."
</item 2>

<item 3>
Al Ulmer
Alfred C. Ulmer, Jr. was born in Jacksonville, Florida in August 1916.

 He was of Swiss extraction on his father's side, his father having been born in Zurich.

 Ulmer graduated from Princeton University in 1939 and joined the United States Navy prior to the start of World War II, ultimately becoming a major head of
intelligence operations during World War II.

 He married Doris Gibson Bridges and had three sons (Alfred III, James and Nicolas) and a daughter (Marguerite).

 He received the Intelligence Medal of Merit when he retired from his position in 1962.

 Ulmer then went on to business and in the 1980s joined the Swiss banking firm Lombard Odier et Cie. in Geneva, later setting up Lombard Odier's operations in
Bermuda.

 He died on June 22, 2000 in Virginia Beach, Virginia.
</item 3>

<item 4>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 4>

<item 5>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 5>

<item 6>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 6>

<item 7>
Ernst Messerschmid
Prof. Dr. Ernst Willi Messerschmid (born May 21, 1945) is a German physicist and former astronaut.
</item 7>

<item 8>
Kaiserjäger (film)
Kaiserjäger is a 1956 Austrian film directed by Willi Forst.
</item 8>

<item 9>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 9>

<item 10>
Winfred Overholser
Winfred Overholser (1892 – October 6, 1964) was an American psychiatrist, president of the American Psychiatric Association, and for 25 years the superintendent
of St. Elizabeths Hospital, a federal institution for the mentally ill in Washington, D.C.
</item 10>
=== END ITEM id=6aa8a197980c9b81 ===

=== ITEM id=5a4ab0680a777395 ===
Gold sentences and facts:
Sentence 0: Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.
  Fact 0.0: Remember the Titans is a 2000 American sports film
  Fact 0.1: Remember the Titans was produced by Jerry Bruckheimer
  Fact 0.2: Remember the Titans was directed by Boaz Yakin
Sentence 1: My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and Ray Walston, based on the 1960s television series of the same name in which Walston starred.
  Fact 1.0: My Favorite Martian is a 1999 American science fiction comedy film
  Fact 1.1: My Favorite Martian stars Christopher Lloyd
  Fact 1.2: My Favorite Martian stars Jeff Daniels
  Fact 1.3: My Favorite Martian stars Daryl Hannah
  Fact 1.4: My Favorite Martian stars Elizabeth Hurley
  Fact 1.5: My Favorite Martian stars Wallace Shawn
  Fact 1.6: My Favorite Martian stars Ray Walston
  Fact 1.7: My Favorite Martian is based on the 1960s television series of the same name
  Fact 1.8: Ray Walston starred in the 1960s television series My Favorite Martian
Sentence 2:  The film grossed $36.8 million domestically against a budget of $65 million.
  Fact 2.0: The film (My Favorite Martian) grossed $36.8 million domestically
  Fact 2.1: The film (My Favorite Martian) had a budget of $65 million

Retrieved items:
<item 1>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 1>

<item 2>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 2>

<item 3>
Honey, I Blew Up the Kid
Honey, I Blew Up the Kid is a 1992 American comedy science fiction film and the sequel to the 1989 film "Honey, I Shrunk the Kids".

 Directed by Randal Kleiser and released by Walt Disney Pictures, the film stars Rick Moranis, Marcia Strassman, Robert Oliveri and Amy O'Neill, who reprise
their roles as Wayne, Diane, Nick, and Amy Szalinski respectively, as well as newcomer Keri Russell as Mandy Park, Nick's love interest and babysitter of Adam,
the Szalinskis' new two-year-old son, whose accidental exposure to Wayne's new industrial-sized growth machine causes him to gradually grow to enormous size.

 Made only three years after "Honey, I Shrunk the Kids", this film is set five years after the events depicted in the previous film.

 It was filmed in 1991.
</item 3>

<item 4>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 4>

<item 5>
Managed Money
Managed Money is a 1934 short comedy film directed by Charles Lamont.

 The film stars Frank Coghlan Jr. and Shirley Temple.

 It was also known as "Frolics of Youth" and "Measured Money".

 This was the second film in which Temple starred as Mary Lou.

 The film tells the story of Sonny and Sid prospecting for gold to pay for their military academy education.

 Sonny's sister, Mary Lou stows away in the back of their car.

 Eventually, they meet an inventor who helps them financially.
</item 5>

<item 6>
My Favorite Martian (film)
My Favorite Martian is a 1999 American science fiction comedy film starring Christopher Lloyd, Jeff Daniels, Daryl Hannah, Elizabeth Hurley, Wallace Shawn and
Ray Walston, based on the 1960s television series of the same name in which Walston starred.

 It was directed by Donald Petrie and written by Sherri Stoner and Deanna Oliver, based on the television series created by John L. Greene.

 Creatures were created by Amalgamated Dynamics from designs by Jordu Schell.

 The film grossed $36.8 million domestically against a budget of $65 million.
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
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 8>

<item 9>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 9>

<item 10>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 10>
=== END ITEM id=5a4ab0680a777395 ===

=== ITEM id=6bdbdb474123f75d ===
Gold sentences and facts:
Sentence 0: Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.
  Fact 0.0: Samantha Cristoforetti was born 26 April 1977 in Milan.
  Fact 0.1: Samantha Cristoforetti is an Italian European Space Agency astronaut.
  Fact 0.2: Samantha Cristoforetti is an Italian Air Force pilot.
  Fact 0.3: Samantha Cristoforetti is an Italian engineer.
Sentence 1: Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force), and a former CNES spationaut.
  Fact 1.0: Jean-Loup Jacques Marie Chrétien was born 20 August 1938.
  Fact 1.1: Jean-Loup Jacques Marie Chrétien is French.
  Fact 1.2: Jean-Loup Jacques Marie Chrétien is retired.
  Fact 1.3: Jean-Loup Jacques Marie Chrétien holds the rank of Général de Brigade (brigadier general).
  Fact 1.4: Jean-Loup Jacques Marie Chrétien served as a general in the Armée de l'Air (French air force).
  Fact 1.5: Jean-Loup Jacques Marie Chrétien is a former CNES spationaut.

Retrieved items:
<item 1>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 1>

<item 2>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 2>

<item 3>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 3>

<item 4>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 4>

<item 5>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 5>

<item 6>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 6>

<item 7>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 7>

<item 8>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 8>

<item 9>
Jean-Loup Puget
Jean-Loup Puget (born 7 March 1947) is a French astrophysicist.

 His current research interests lie in the Cosmic Microwave Background.

 Jean-Loup Puget and his collaborators reported the first identification of the Cosmic infrared background using COBE data.

 He is also, along with Alain Léger, credited with the origin of the hypothesis that the series of infrared lines observed in numerous astrophysical objects are
caused by emission from polycyclic aromatic hydrocarbons.

 He is currently principal investigator of the HFI module of the Planck space mission.
</item 9>

<item 10>
Soyuz T-6
Soyuz T-6 was a manned spaceflight to Earth orbit to the Salyut 7 space station in 1982.

 Along with two Soviet cosmonauts, the crew included a Frenchman, Jean-Loup Chrétien.
</item 10>
=== END ITEM id=6bdbdb474123f75d ===
