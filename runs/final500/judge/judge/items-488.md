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

This file has 13 items.

=== ITEM id=1152f033e37e2374 ===
Gold sentences and facts:
Sentence 0: Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.
  Fact 0.0: Buried Country is a highly regarded documentary film
  Fact 0.1: Buried Country is a highly regarded book
  Fact 0.2: Buried Country is a highly regarded soundtrack album
  Fact 0.3: Buried Country is now also a stageshow
Sentence 1: South of the Border is a 2009 American documentary film directed by Oliver Stone.
  Fact 1.0: South of the Border is a documentary film
  Fact 1.1: South of the Border is American
  Fact 1.2: South of the Border is from 2009
  Fact 1.3: South of the Border was directed by Oliver Stone

Retrieved items:
<item 1>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 1>

<item 2>
Love Birds (musical)
Love Birds is an original musical with book, music and lyrics by Robert J. Sherman.

 The musical officially premiered at the 2015 Edinburgh Fringe Festival at The Pleasance on August 7, 2015.

 The production was directed and choreographed by Stewart Nicholls.

 Prior to the Edinburgh run, there were two previews held in London at the Lost Theatre (in Stockwell) on July 28 and 29.

 The Original Edinburgh Cast Recording was recorded on July 30 and mixed and mastered on July 31 and August 3 and was first released by SimG Records on August
12 at the Edinburgh Festival.

 Two Edinburgh previews took place on August 5 and 6 with Press Night taking place on August 8.

 The final festival performance took place on August 31.

 The show starred Ruth Betteridge, Greg Castiglioni, John Guerrasio, George Knapper, Jonny Purchase, Joanna Sawyer, Anna Stolli, Rafe Watts and Ryan Willis.
</item 2>

<item 3>
Cinema of Uruguay
The cinema of Uruguay has a role in the culture of Uruguay and is a part of Latin American cinema.

 Since the late 1990s, the Uruguayan cinema undergoes a process of evolution, during which its films have received positive reviews and been internationally
recognized.

 Over 120 films, fictions and non-fictions, have been produced since then.
</item 3>

<item 4>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 4>

<item 5>
The Colleen Bawn
The Colleen Bawn, or The Brides of Garryowen is a melodramatic play written by Irish playwright Dion Boucicault.

 It was first performed at Miss Laura Keene's Theatre, New York, on 27 March 1860 with Laura Keene playing Anne Chute and Boucicault playing Myles na Coppaleen.

 It was most recently performed in Dublin at the Project Arts Centre in July and August 2010.

 Several film versions have also been made.
</item 5>

<item 6>
Plain Clothes Theatre Productions
Plain Clothes Theatre Productions is a Bristol-based theatre company producing contemporary plays from around the globe.

 Formed in 2003 by artistic director Sam Berger, the company has produced work in London, Bristol, Cheltenham, Toronto and Vancouver.

 Their work has included plays by Joe Orton, Laura Wade and David Mamet, and their style is based around the teachings of American acting coach Sanford Meisner.
</item 6>

<item 7>
Buried Country
Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.

 It is a prosopography, created by noted writer Clinton Walker, that tells the story of Australian country music in the Aboriginal community by focussing on the
genre's most important stars.
</item 7>

<item 8>
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
</item 8>

<item 9>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 9>

<item 10>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 10>
=== END ITEM id=1152f033e37e2374 ===

=== ITEM id=f3893044ac61d3b9 ===
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
Frank A. Armstrong
Frank Alton Armstrong Jr. (May 24, 1902 – August 20, 1969) was a lieutenant general of the United States Air Force.

 As a brigadier general in the United States Army Air Forces during World War II, he was the inspiration for the main character in the novel and subsequent
film, "Twelve O'Clock High."

 After the war, he held a variety of senior leadership positions prior to and following the establishment of the USAF as an independent service in 1947.

 Promoted to major general in 1950, he advanced to lieutenant general in 1956 and retired at that rank in 1962.
</item 1>

<item 2>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 2>

<item 3>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 3>

<item 4>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 4>

<item 5>
Myron F. Diduryk
Myron F. Diduryk (July 15, 1938 to April 24, 1970) was a Ukrainian-American United States Army Major, who played a key role as an infantry company commander in
the Battle of Ia Drang, the first major battle of the Vietnam War.

 His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".

 Moore said that Diduryk was, “… the finest battlefield company commander I had ever seen, bar none.”

 Diduryk was killed in action on his second tour in Vietnam.
</item 5>

<item 6>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
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
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 9>

<item 10>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 10>
=== END ITEM id=f3893044ac61d3b9 ===

=== ITEM id=100359947cbdd75f ===
Gold sentences and facts:
Sentence 0: Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central business district of Brainerd, a city in Crow Wing County, Minnesota, United States.
  Fact 0.0: Brainerd Lakes Regional Airport has IATA code BRD.
  Fact 0.1: Brainerd Lakes Regional Airport has ICAO code KBRD.
  Fact 0.2: Brainerd Lakes Regional Airport has FAA LID BRD.
  Fact 0.3: Brainerd Lakes Regional Airport is a public use airport.
  Fact 0.4: Brainerd Lakes Regional Airport is located three nautical miles (6 km) northeast of the central business district of Brainerd.
  Fact 0.5: Brainerd is a city in Crow Wing County, Minnesota, United States.
Sentence 1: Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City, Utah in the United States.
  Fact 1.0: Salt Lake City International Airport has IATA code SLC.
  Fact 1.1: Salt Lake City International Airport has ICAO code KSLC.
  Fact 1.2: Salt Lake City International Airport has FAA LID SLC.
  Fact 1.3: Salt Lake City International Airport is a civil-military airport.
  Fact 1.4: Salt Lake City International Airport is located about 4 miles west of Downtown Salt Lake City.
  Fact 1.5: Downtown Salt Lake City is in Utah, United States.

Retrieved items:
<item 1>
Green Line (TRAX)
The Green Line is a light rail line on the Utah Transit Authority's (UTA) TRAX system in Salt Lake City, Utah, in the United States, operated by the Utah
Transit Authority (UTA).

 It opened on August 7, 2011, and runs between Airport Station at the Salt Lake City International Airport and West Valley Central Station in West Valley City
(via Downtown Salt Lake City) serving a total of eighteen stations: thirteen in Salt Lake City, one in South Salt Lake, and four in West Valley City.
</item 1>

<item 2>
WLMR
WLMR (1450 AM) is a radio station broadcasting a religious format.

 Licensed to Chattanooga, Tennessee, USA, the station serves the Chattanooga area.

 The station is currently owned by Wilkins Communications Network, Inc. and features programming from USA Radio Network.

 In the early 1980s, the station was automated and played country music from studios in a strip mall on Brainerd Rd.

 It was also Chattanooga's first talk radio station when it held the WZRA call letters.

 WZRA was the first home of Jeff Styles, and also featured well known personalities such as Kelly McCoy and Robert T. Nash.
</item 2>

<item 3>
Temple Square (UTA station)
Temple Square is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 3>

<item 4>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 4>

<item 5>
City Center (UTA station)
City Center is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 5>

<item 6>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 6>

<item 7>
North Temple Bridge/Guadalupe (UTA station)
North Temple Bridge/Guadalupe is a light rail and commuter rail station in Salt Lake City, Utah, United States served by the Green Line of Utah Transit
Authority's (UTA) TRAX light rail system and by the "FrontRunner", UTA's commuter rail train that operates along the Wasatch Front with service from Pleasant
View in northern Weber County through Ogden, Davis County, Salt Lake City, and Salt Lake County to Provo in central Utah County.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City) and connects with the rest of
the TRAX system and UTA's S Line streetcar.

 Although the official name of the station is "North Temple Bridge/Guadalupe", it often referenced by UTA as simply North Temple.
</item 7>

<item 8>
Salt Lake City Air Route Traffic Control Center
Salt Lake City Air Route Traffic Control Center (ZLC) (radio communications "Salt Lake Center") is one of 22 FAA Area Control Centers in the United States.

 It is located in Salt Lake City, Utah, adjacent to Salt Lake City International Airport.

 It was opened in 1939 and was originally located on the third floor of the old Salt Lake City International Airport terminal.

 The Salt Lake Center (ZLC) covers one of the largest geographical areas of any other control center, totaling approximately 350,000 squares miles.
</item 8>

<item 9>
Brunswick Golden Isles Airport
Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five
nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.

 It is mostly used for general aviation, but is also served by one commercial airline.
</item 9>

<item 10>
Arena (UTA station)
Arena is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit Authority's
(UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line has service from the Salt Lake City International Airport and to West Valley City via Downtown Salt Lake City.
</item 10>
=== END ITEM id=100359947cbdd75f ===

=== ITEM id=a032a8584e4c145f ===
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
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 3>

<item 4>
Jarrett Hurd
Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
</item 4>

<item 5>
Scott Dixon (boxer)
featherweight to 167+1/2 lb , i.e. Super middleweight.
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
Nedal Hussein
Nedal "Skinny" Hussein (born 1 December 1977) is an Australian professional bantam/super bantam/feather/super feather/Lightweight boxer of the 1990s and 2000s
who won the Australian super bantamweight title, Australian bantamweight title, International Boxing Federation (IBF) Pan Pacific featherweight title, World
Boxing Federation (WBF) featherweight title, World Boxing Union (WBU) super bantamweight title, World Boxing Organization (WBO) Asia Pacific super featherweight
title, International Boxing Federation (IBF) Pan Pacific super featherweight title, International Boxing Organization (IBO) Inter-Continental super
featherweight title, and Commonwealth super bantamweight title, and was a challenger for the World Boxing Council (WBC) International super bantamweight title
against Manny Pacquiao, World Boxing Council (WBC) super bantamweight title against Óscar Larios, World Boxing Organization (WBO) featherweight title against
Scott Harrison, Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki,
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
Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 He is the former UFC Light Heavyweight Champion, UFC 12 Heavyweight Tournament Champion, and the Cage Rage World Light Heavyweight Champion.

 As of June 16, 2017, he is the #11 contender in official UFC Middleweight rankings.
</item 9>

<item 10>
Richard Williams (boxer)
Richard "The Secret" Williams ( (1971--) 09 1971 (age (2017)-(1971)-((11)<(05)or(11)==(05)and(30)<(09)) ) is an English professional light middle/middle/super
middleweight boxer of the 1990s, 2000s and 2010s who won the International Boxing Organization (IBO) light middleweight title, and Commonwealth light
middleweight title (twice), and was a challenger for the International Boxing Organization (IBO) light middleweight title against Sergio Gabriel Martinez, and
British Boxing Board of Control (BBBofC) British middleweight title against Howard Eastman, his professional fighting weight varied from 149+3/4 lb , i.e. light
middleweight to 162+1/4 lb , i.e. super middleweight.

 Richard Williams is trained by Brian Lawrence, and managed by Barry Hearn, and John Rooney.
</item 10>
=== END ITEM id=a032a8584e4c145f ===

=== ITEM id=9c19e875c1cff097 ===
Gold sentences and facts:
Sentence 0: Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
  Fact 0.0: Kelechukwu 'Kele' Okereke was born on 13 October 1981.
  Fact 0.1: Kelechukwu 'Kele' Okereke is also known by the mononym Kele.
  Fact 0.2: Kelechukwu 'Kele' Okereke is a British musician.
  Fact 0.3: Kelechukwu 'Kele' Okereke is best known as the lead singer and rhythm guitarist of the indie rock band Bloc Party.
Sentence 1: Jon Hume (born 6 November 1983) is the lead singer of Evermore.
  Fact 1.0: Jon Hume was born on 6 November 1983.
  Fact 1.1: Jon Hume is the lead singer of Evermore.
Sentence 2:  Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
  Fact 2.0: Jon is the eldest of the three brothers.
  Fact 2.1: The three brothers make up the New Zealand band Evermore.
  Fact 2.2: Evermore is a New Zealand band.

Retrieved items:
<item 1>
Graph query row:
entity: Jon Hume
relationship: MEMBER_OF
related: Evermore
</item 1>

<item 2>
Evermore (Evermore album)
Evermore is the self-titled international debut album by New Zealand rock band Evermore, which was also released in Australia and New Zealand on 12 March 2010.

 The album is a collection of songs from the band's entire history, as well as three new songs including the single "Underground".
</item 2>

<item 3>
Dann Hume
Dann Hume (born Daniel Benjamin Cobbe, 1 September 1987, Whangaparaoa, New Zealand) is a singer-songwriter and record producer.

 Hume is the youngest of three brothers (with Peter Hume and Jon Hume) who make up the alternative rock band, Evermore since 1999.
</item 3>

<item 4>
Between the Lines (Evermore song)
"Between the Lines" is the first single by Evermore, taken from their third studio album "".

 Evermore's Jon Hume said that "Between the Lines was the first song to come out of a search for a new musical experience as a band."

 It was released as a free download on Evermore's official website on 10 November 2008.
</item 4>

<item 5>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 5>

<item 6>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 6>

<item 7>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 7>

<item 8>
Jon Hume
Jon Hume (born 6 November 1983) is the lead singer of Evermore.

 He is a singer-songwriter, multi-instrumentalist, Record Producer and Studio owner.

 Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
</item 8>

<item 9>
Kele Okereke
Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of
the indie rock band Bloc Party.
</item 9>

<item 10>
Running (Evermore song)
"Running" is the lead single by alternative rock band Evermore, taken from their second studio album, "Real Life" (July 2006).

 It was written by backing vocalist and drummer, Dann Hume.

 It was issued on 3 June, ahead of the album.

 It was co-produced by Jon with John Alagía (Dave Matthews Band, John Mayer).

 It peaked at No. 4 on their native RIANZ Singles Chart in New Zealand and No. 5 on their adopted country of Australia's ARIA Singles Chart.
</item 10>
=== END ITEM id=9c19e875c1cff097 ===

=== ITEM id=e38c1c2eaed9cdca ===
Gold sentences and facts:
Sentence 0: Chester Erskine (November 29, 1905 – April 7, 1986) was a Hollywood and Broadway director, writer, and producer.
  Fact 0.0: Chester Erskine was born on November 29, 1905.
  Fact 0.1: Chester Erskine died on April 7, 1986.
  Fact 0.2: Chester Erskine was a Hollywood and Broadway director.
  Fact 0.3: Chester Erskine was a Hollywood and Broadway writer.
  Fact 0.4: Chester Erskine was a Hollywood and Broadway producer.
Sentence 1: Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.
  Fact 1.0: Sam Taylor was born on August 13, 1895, in New York City.
  Fact 1.1: Sam Taylor died on March 6, 1958, in Santa Monica.
  Fact 1.2: Sam Taylor was a film director.
  Fact 1.3: Sam Taylor was a screenwriter.
  Fact 1.4: Sam Taylor was a film producer.
  Fact 1.5: Sam Taylor was most active in the silent film era.

Retrieved items:
<item 1>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 1>

<item 2>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 2>

<item 3>
Chester Erskine
Chester Erskine (November 29, 1905 – April 7, 1986) was a Hollywood and Broadway director, writer, and producer.
</item 3>

<item 4>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 4>

<item 5>
Mavis Taylor
Mavis Taylor (1915 – 14 March 2007) was an Australian who was named an Australian Living Treasure for her humanitarian work for the people of East Timor in her
later years.
</item 5>

<item 6>
Speedy West
Wesley Webb West (January 25, 1924 – November 15, 2003), better known as Speedy West, was an American pedal steel guitarist and record producer.

 He frequently played with Jimmy Bryant, both in their own duo and as part of the regular Capitol Records backing band for Tennessee Ernie Ford and many others.

 He also played on Loretta Lynn's first single.
</item 6>

<item 7>
Mary Elizabeth Bliss
Mary Elizabeth "Betty" Taylor Bliss Dandridge, born Mary Elizabeth Taylor (April 20, 1824 – July 25, 1909), was the youngest of the three surviving daughters of
President Zachary Taylor (1849-1850) and Margaret Mackall Smith Taylor.
</item 7>

<item 8>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 8>

<item 9>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 9>

<item 10>
Dick Herrick
The Very Rev Richard William Herrick (3 December 1913 - 5 May 1981) was an eminent Anglican priest in the 20th century.

 He was educated at King Edward VI School Retford and Leeds University and was initially a civil servant.

 He was ordained after a period of study at the College of the Resurrection, Mirfield in 1940.

 he held curacies at Duston and Portsea, Portsmouth before being appointed Vicar of St Michael’s, Northampton in 1947, a post he held for a decade.

 He was then a Canon Residentiary of Chelmsford Cathedral until 1978 when he was appointed Provost of Chelmsford.

 He died in post.
</item 10>
=== END ITEM id=e38c1c2eaed9cdca ===

=== ITEM id=46f5c1c289df3777 ===
Gold sentences and facts:
Sentence 0: Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.
  Fact 0.0: Paul Clyne was the District Attorney of Albany County, New York.
  Fact 0.1: Paul Clyne held this office from January 2001.
  Fact 0.2: Paul Clyne held this office through December 2004.
Sentence 1: P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.
  Fact 1.0: P. David Soares was born on October 26, 1969.
  Fact 1.1: P. David Soares was born in Brava, Cape Verde.
  Fact 1.2: P. David Soares is the District Attorney of Albany County, New York.

Retrieved items:
<item 1>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 1>

<item 2>
City Cruises
City Cruises is a public limited company that operates scheduled public sightseeing cruises on the River Thames in London, serving Westminster, London Eye,
Tower and Greenwich piers every day throughout the year.

 It operates the services mainly using purpose-built wheelchair-accessible 'RiverLiner' vessels, which have a large open top deck and enclosed lower saloon with
panoramic windows and bars, backed up by some ex-Paris single-deck conservatory-style sightseeing boats.

 The company also owns a number of other boats with which it provides a variety of private charter and waterage services on the River Thames.

 City Cruises owns the London Showboat brand, under which it operates an evening dinner, cabaret and dance cruise from Westminster Pier throughout the year, and
also owns and operates the Restaurant Ship Hispaniola, which is permanently moored alongside Victoria Embankment close to Embankment Underground station in
central London.
</item 2>

<item 3>
Skip Bolen
Skip Bolen is a Southern photographer of musicians, architecture, lifestyle and the culture of New Orleans.

 Born in Lafayette, Louisiana, he moved to New Orleans where he began his publishing career as a designer and art director.

 After moving to New York City, he began working at "House & Garden," renamed "HG," as Senior Designer in January 1988 with Anna Wintour and Alexander Liberman
at Condé Nast Publications.

 Spending evenings in jazz clubs, he began photographing jazz musicians in New York City and often when he regularly returned to New Orleans.

 After three years at Condé Nast Publications in New York City, he returned to New Orleans to pursue his jazz photography full-time.

 In 1998, he moved to Los Angeles where he became art director of House of Blues for seven years while photographing at night and weekends.

 He continued photographing jazz musicians and had his first major solo exhibition at the Jazz Bakery in Los Angeles on August 9, 2002.

 On July 4, 2006, he returned to New Orleans to pursue photography full-time documenting the recovery and rebuilding of New Orleans since Hurricane Katrina,
documenting the jazz scene, night-time photography and other photographic projects.
</item 3>

<item 4>
David Soares
P. David Soares (born October 26, 1969, Brava, Cape Verde) is the Albany County, N.Y. District Attorney.

 He is a Democrat.
</item 4>

<item 5>
Paul Soniat
Paul Soniat is the Director of the New Orleans City Park Botanical Gardens.

 He was born at the old Touro Hospital in New Orleans.

 His family has been in New Orleans since 1727 and he grew up and later lived on the street bearing his family name.

 Paul is a self-taught piano player, and has released two CDs, absorbing the sights, sounds, and flavor of New Orleans.

 His first CD, titled "Born in New Orleans", was released in April 2005.

 His second CD was released after Hurricane Katrina hit the Gulf coast and the New Orleans' levees failed.

 This second CD is appropriately titled "Below the Water Line".
</item 5>

<item 6>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 6>

<item 7>
Laurent Manrique
Laurent Manrique (born March 30, 1966) is a French restaurateur and Michelin-starred chef.

 Currently overseeing projects in New York and San Francisco, Laurent has extensive professional experience in both locales.

 He was the Corporate Executive Chef of Aqua, an upscale seafood restaurant in San Francisco, where he earned 2 Michelin stars in 2006, the first time Michelin
came to the San Francisco Bay Area, and again in 2007 and 2008.

 Upon his departure in 2009, the restaurant lost its Michelin status.

 For several years, he has overseen the Cafe de la Presse, located on the corner of Bush Street and Grant Avenue, as well as two wine bars: Blanc et Rouge and
Rouge and Blanc.

 In New York, he recently opened Millesime, a seafood brasserie in the Carlton Hotel.
</item 7>

<item 8>
Paul Clyne
Paul Clyne was the District Attorney of Albany County, New York from January 2001 through December 2004.

 A graduate of Albany Law School, he spent about 14 years as an assistant district attorney, before he was tapped by local politicians to replace the retiring
District Attorney, Sol Greenberg.

 He was defeated for re-election by David Soares, first in the Democratic Party primary election in September 2004, and then in the general election in November
2004, in which he ran on an independent line.

 After a stint teaching at the New York Prosecutors Institute, he went into private practice as a criminal defense attorney in 2007, with an office in Albany,
New York.
</item 8>

<item 9>
Stewart Spiers
Stewart Spiers was a small but innovative firm of plane-makers in Scotland, founded first of all in Ayr in Ayrshire and continuing under the registered name of
Stewart Speirs Ltd ["sic"] in Paisley, Renfrewshire, from c. 1933 until its demise in the mid to late 1930s.

 Like the Glasgow firm of Alexander Mathieson & Sons, Spiers benefited hugely from the thriving industries on the Firth of Clyde in the latter half of the
nineteenth century.
</item 9>

<item 10>
Richard Pleasance
Richard Arnold Pleasance is an Australian rock musician and producer.

 He was a founding member of Boom Crash Opera on guitar, bass guitar, vocals and as a songwriter in 1985; they released three albums before Pleasance left in
1992.

 Their hit Australian Recording Industry Association (ARIA) singles, "Great Wall" (No. 5, 1986) and "Onion Skin" (No. 11, 1989) were co-written by Pleasance,
who also co-produced their second album, "These Here Are Crazy Times" (No. 10, 1989).

 His debut solo release, "Galleon" received four nominations at the ARIA Music Awards for 1992.

 Pleasance composed the theme music for Australian television series, "SeaChange" (1998–2001), for 2006 feature film "Kenny" and more recently he composed the
theme music for the prison drama series "Wentworth".

 Pleasance is married to Michelle and, as from May 2009, he was living in Hepburn Springs, Victoria where he has a recording studio.
</item 10>
=== END ITEM id=46f5c1c289df3777 ===

=== ITEM id=70a310b9a2cdc3ed ===
Gold sentences and facts:
Sentence 0: Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.
  Fact 0.0: Yucca is a genus of perennial shrubs and trees.
  Fact 0.1: Yucca is in the family Asparagaceae.
  Fact 0.2: Yucca is in the subfamily Agavoideae.
Sentence 1: Rodgersia is a genus of flowering plants in the Saxifragaceae family.
  Fact 1.0: Rodgersia is a genus of flowering plants.
  Fact 1.1: Rodgersia is in the Saxifragaceae family.
Sentence 2:  "Rodgersia" are herbaceous perennials originating from east Asia.
  Fact 2.0: Rodgersia are herbaceous perennials.
  Fact 2.1: Rodgersia originate from east Asia.

Retrieved items:
<item 1>
Yucca
Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.

 Its 40-50 species are notable for their rosettes of evergreen, tough, sword-shaped leaves and large terminal panicles of white or whitish flowers.

 They are native to the hot and dry (arid) parts of the Americas and the Caribbean.

 Early reports of the species were confused with the cassava ("Manihot esculenta").

 Consequently, Linnaeus mistakenly derived the generic name from the Taíno word for the latter, "yuca" (spelled with a single "c").

 It is commonly found growing in rural graveyards and when in bloom the cluster of (usually pale) flowers on a thin stalk appear as floating apparitions.
</item 1>

<item 2>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 2>

<item 3>
Yucca glauca
Yucca glauca (syn.

 "Yucca angustifolia") is a species of perennial evergreen plant, adapted to xeric (dry)growth conditions.

 It is also known as small soapweed, soapweed yucca, Spanish bayonet, Great Plains yucca and beargrass.
</item 3>

<item 4>
Yucca jaliscensis
Yucca jaliscensis is a "Yucca" species native to the highlands of southwestern Mexico.

 Common names for this species include Jalisco Yucca, Jalisco Soapwort, Izote Yucca.

 It is native to mountainous areas at about 5000 feet in the States of Jalisco, Colima and Guanajuato.
</item 4>

<item 5>
Yucca rupicola
Yucca rupicola is a plant in the family Asparagaceae, known as the twistleaf yucca, twisted-leaf yucca, Texas yucca or twisted-leaf Spanish-dagger.

 The species was described by George Heinrich Adolf Scheele in 1850.

 This is a small, acaulescent plant with distinctive twisted leaves.

 It is native to the Edwards Plateau region of Texas and also to northeastern Mexico (Coahuila, Nuevo León).
</item 5>

<item 6>
Yucca × schottii
Yucca" × "schottii is a plant species in the genus "Yucca", native to southern Arizona, southwestern New Mexico, and the northern parts of Sonora and Chihuahua.

.

 The common names are Schott's yucca, hoary yucca, and mountain yucca.

 The "×" in the name indicates that this is a nothospecies, regarded as being a natural hybrid between two other species.

 In this case, "Yucca" × "schottii" is believed to have originated as a hybrid between "Y. baccata" and "Y. madrensis".

 "Yucca" × "schottii" is firmly established and does reproduce freely in the wild.
</item 6>

<item 7>
Yucca gloriosa var. tristis
Yucca gloriosa" var.

 "tristis (syn.

 Yucca recurvifolia, "Yucca gloriosa" var.

 "recurvifolia"), known as curve-leaf yucca, curved-leaved Spanish-dagger or pendulous yucca, is a variety of "Yucca gloriosa".

 It is often grown as an ornamental plant, but is native to the southeastern United States.

 In contrast to "Y. gloriosa" var.

 "tristis", the leaves of "Y. gloriosa" var.

 "gloriosa" are hard stiff, erect and narrower.
</item 7>

<item 8>
Rodgersia
Rodgersia is a genus of flowering plants in the Saxifragaceae family.

 "Rodgersia" are herbaceous perennials originating from east Asia.
</item 8>

<item 9>
Yucca brevifolia
Yucca brevifolia is a plant species belonging to the genus "Yucca".

 It is tree-like in habit, which is reflected in its common names: Joshua tree, yucca palm, tree yucca, and palm tree yucca.
</item 9>

<item 10>
Hesperoyucca whipplei
Hesperoyucca whipplei (syn.

 "Yucca whipplei" ) (chaparral yucca, our Lord's candle, Spanish bayonet, Quixote yucca or foothill yucca is a species of flowering plant closely related to,
and formerly usually included in, the genus "Yucca".

 It is native to southern California, United States and Baja California, Mexico, where it occurs mainly in chaparral, coastal sage scrub, and oak woodland plant
communities at altitudes of 0–2500 m.
</item 10>
=== END ITEM id=70a310b9a2cdc3ed ===

=== ITEM id=57eda8759d18d448 ===
Gold sentences and facts:
Sentence 0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
  Fact 0.0: The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia.
  Fact 0.1: The 1997 European Athletics Junior Championships were held on July 24-27.
Sentence 1: Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.
  Fact 1.0: Ljubljana is the capital of Slovenia.
  Fact 1.1: Ljubljana is the largest city of Slovenia.
  Fact 1.2: Ljubljana is also known by other, historical names.

Retrieved items:
<item 1>
1964 European Junior Games
The 1964 European Junior Games was the first edition of the biennial athletics competition for European athletes aged under twenty.

 It was an unofficial competition without sanction from the European Athletic Association.

 The event was held at the 10th-Anniversary Stadium in Warsaw, Poland.

 The competition eventually led to the creation of the European Athletics Junior Championships in 1970.
</item 1>

<item 2>
1994 Athens International – Doubles
Horacio de la Peña and Jorge Lozano were the defending champions, but did not participate this year.
</item 2>

<item 3>
2015 European Athletics Junior Championships
The 2015 European Athletics Junior Championships were the 23rd edition of the biennial European U20 athletics championships.

 They were held in Eskilstuna, Sweden from 16 July to 19 July.
</item 3>

<item 4>
1997 European Athletics Junior Championships
The 1997 European Athletics Junior Championships were held in Ljubljana, Slovenia on July 24–27.
</item 4>

<item 5>
European Athletics U20 Championships
The European Athletics U20 Championships (formerly named the European Athletics Junior Championships up to 2015) are the European championships for athletes who
are 19 years of age or under, which is the age range recognised by the IAAF as junior athletes.

 The event is currently organized by the European Athletic Association.
</item 5>

<item 6>
1968 European Junior Games
The 1968 European Junior Games was the third edition of the biennial athletics competition for European athletes aged under twenty.

 It was the final edition of the competition under that name and was succeeded by the European Athletics Junior Championships in 1970.

 The event was held in Leipzig, part of the German Democratic Republic at that time.

 The competition was mainly between Eastern European countries, with Belgium, Austria, Netherlands and Greece being present from the west.
</item 6>

<item 7>
Amela Terzić
Amela Terzić (Serbian Cyrillic: Амела Терзић, born 2 January 1993 in Priboj, FR Yugoslavia) is a Serbian middle-distance runner.

 She won two gold medals at 2011 European Athletics Junior Championships in Tallinn and was the junior champion at the 2012 European Cross Country
Championships.

 She has also been a medallist in the 1500 metres at the World Junior and World Youth Championships and a gold medalist at the 2013 European U23 Championship.
</item 7>

<item 8>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 8>

<item 9>
Athletics at the 2002 Asian Games
Athletics was contested from October 7 to October 14 at the 2002 Asian Games in Busan Asiad Main Stadium, Busan, South Korea.

 A total of 460 athletes (comprising 281 men and 179 women) from 39 nations took part in the competition.

 Afghanistan, Bhutan, Brunei, Laos and Maldives were the only nations without a representative in the events.
</item 9>

<item 10>
Ljubljana
Ljubljana (] , ] ; also known by other, historical names) is the capital and largest city of Slovenia.

 It has been the cultural, educational, economic, political, and administrative center of independent Slovenia since 1991.

 Its central geographic location within Slovenia, transport connections, concentration of industry, scientific and research institutions, and cultural tradition
are contributing factors to its leading position.
</item 10>
=== END ITEM id=57eda8759d18d448 ===

=== ITEM id=ddc424ef25344507 ===
Gold sentences and facts:
Sentence 0: Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
  Fact 0.0: Hänt Extra is a Swedish language magazine.
  Fact 0.1: Hänt Extra is a weekly magazine.
  Fact 0.2: Hänt Extra is a celebrity and youth magazine.
  Fact 0.3: Hänt Extra is published by Aller Media.
  Fact 0.4: Hänt Extra is published in Stockholm, Sweden.
Sentence 1: Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
  Fact 1.0: Tea Moderna is a Macedonian woman's weekly magazine.
  Fact 1.1: Tea Moderna's name in Macedonian is Теа Модерна.
  Fact 1.2: Tea Moderna is a magazine for women.

Retrieved items:
<item 1>
Den Za Nas
Den Za Nas (in Macedonian: Ден за нас ) is the title of the fourth Macedonian language album by the Macedonian superstar Toše Proeski released in 2004.

 It is a 2-CD set which contains all of the songs performed on the National final in order to select a song for the Eurovision Song Contest 2004 with which
Proeski would represent Macedonia in the contest.

 The winning song was "Life" which qualified for the final and finished on the 14th place in the final.

 Some extra songs such as "Dzvezdo Severnice", "Me Sakaš So Zborovi", "Ima Li Den Za Nas" and others were recorded especially for the album.
</item 1>

<item 2>
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 2>

<item 3>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 3>

<item 4>
The entries profile publishers, periodicals, and literary figures: Random House imprints Villard and Simon & Schuster; Macedonian weekly *Makedonsko Sonce*;
Nepalese regent and pioneering author Queen Tripurasundari; print-on-demand firm Mimeo; stamp reference Minkus; radical Black British publisher
Bogle-L’Ouverture; spiritual media company Sounds True; historical reprint specialist Applewood; podsafe music network PMN; speculative-fiction zine *LC
</item 4>

<item 5>
Profiles span journalists, artists, politicians, scholars and public figures: Nikola Mladenov founded Macedonia’s respected Fokus and inspired an
investigative-journalism award; Johannes Grenzfurthner leads art group monochrom; François Genoud financed Nazi networks and Middle Eastern militants; Edmund de
Unger collected Islamic art; Adel Ferdosipour hosts Iran’s Navad; Hans Aschenborn painted African wildlife; Katarina Ivanovska models and acts. Nicolae
Titulescu
</item 5>

<item 6>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 6>

<item 7>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 7>

<item 8>
2015 Macedonian Football Supercup
The 2015 Macedonian Supercup was the 3rd Macedonian Football Supercup, an annual Macedonian football match played between the winners of the previous season's
First League and Macedonian Cup.

 The game will be played between Rabotnički, who beat Teteks to win the 2015 Macedonian Cup Final, and Vardar, champions of the 2014–15 First League.

 It was played at Philip II Arena, Skopje and was won by Vardar by 4–3 on penalties after the match was ended 1–1 after the regular and extra time.
</item 8>

<item 9>
Nikola Mladenov
Nikola Mladenov (Macedonian: Никола Младенов ; 10 March 1964 – 26 March 2013) was a Macedonian journalist, actor, founder and managing editor of the weekly
newspaper "Fokus", widely regarded as one of the most reputable journalists in Macedonia, who along with his colleagues from the newspaper "Mlad borec" is
considered to have laid the basis for the pluralistic principles in the Macedonian journalism.

 The "Award for best research story of the year – Nikola Mladenov", which is awarded by the Macedonian Institute for Media, is named after him.
</item 9>

<item 10>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 10>
=== END ITEM id=ddc424ef25344507 ===

=== ITEM id=1210a867a7c40c71 ===
Gold sentences and facts:
Sentence 0: Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.
  Fact 0.0: Mulled wine is a beverage.
  Fact 0.1: Mulled wine is usually made with red wine.
  Fact 0.2: Mulled wine is usually made with various mulling spices.
  Fact 0.3: Mulled wine is sometimes made with raisins.
Sentence 1:  It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.
  Fact 1.0: Mulled wine is served hot or warm.
  Fact 1.1: Mulled wine is alcoholic.
  Fact 1.2: There are non-alcoholic versions of mulled wine.
Sentence 2: Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
  Fact 2.0: Blue Lagoon is a popular summer cocktail.
  Fact 2.1: Blue Lagoon features blue Curaçao.

Retrieved items:
<item 1>
Feuerzangenbowle
Feuerzangenbowle (] ) is a traditional German alcoholic drink for which a rum-soaked sugarloaf is set on fire and drips into mulled wine.

 It is often part of a Christmas or New Year's Eve tradition.

 The name translates literally to "fire-tongs punch".
</item 1>

<item 2>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 2>

<item 3>
Mulled wine
Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.

 It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.

 It is a traditional drink during winter, especially around Christmas.
</item 3>

<item 4>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 4>

<item 5>
Mulling spices
Mulling spices is a spice mixture used in drink recipes.

 The spices are usually used to spice hot apple cider, mulled wine, wassail, and other drinks (such as juices) during the autumn or winter.

 A "mulled" drink is a drink which has been prepared with these spices (usually through heating the drink in a pot with mulling spices and then straining).

 Mulling spices may also be added to the brewing process to make spiced beer.
</item 5>

<item 6>
Blue Lagoon (cocktail)
Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
</item 6>

<item 7>
Blue Lagoon: The Awakening
Blue Lagoon: The Awakening is a 2012 American made-for-television film that premiered on Lifetime on June 16, 2012.

 Indiana Evans and Brenton Thwaites star in the film, which is based on the novel "The Blue Lagoon" and its previous film adaptations.

 Christopher Atkins, the male lead of the 1980 film "The Blue Lagoon", also appears in the film.
</item 7>

<item 8>
The Blue Lagoon (novel)
The Blue Lagoon is a romance novel written by Henry De Vere Stacpoole and was first published by T. Fisher Unwin in 1908.

 It is the first novel of the "Blue Lagoon" trilogy, which also includes "The Garden of God" (1923) and "The Gates of Morning" (1925).

 The novel has inspired several film adaptations, most notably "The Blue Lagoon" starring Brooke Shields as Emmeline and Christopher Atkins as Richard ("Dicky"
in the book).
</item 8>

<item 9>
Oz and Hugh Drink to Christmas
Oz and Hugh Drink to Christmas is a BBC television programme in which wine personality Oz Clarke and comedian Hugh Dennis travel through Britain to sample a
wide array of seasonal Christmas beverages, including whisky, winter ales, mulled wine, wassail, sloe gin, Buck's Fizz, Port wine and Sherry.

 Upon its 20 December 2009 broadcast on BBC Two, it had a viewership of approximately 2.4 million with an audience share of 9%.
</item 9>

<item 10>
The entries cover foods, drinks, customs, and folklore. Apple dumplings are baked pastry-wrapped apples; mulling spices flavor warm seasonal drinks such as
mulled wine and Finnish glögi. Cocktails include Flaming Volcano (a rum-and-brandy tropical group drink served in an ignited volcano bowl), Flaming Doctor
Pepper, Apple-Kneel, and Blue Lagoon; flaming drinks use lit high-proof alcohol for spectacle or flavor. Other topics include Finland’s Christmas feast, the
fire
</item 10>
=== END ITEM id=1210a867a7c40c71 ===

=== ITEM id=086cb3c6e3a3bd2c ===
Gold sentences and facts:
Sentence 0: Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.
  Fact 0.0: Jürgen Melzer was born on 22 May 1981.
  Fact 0.1: Jürgen Melzer was born in Vienna.
  Fact 0.2: Jürgen Melzer is an Austrian tennis player.
Sentence 1: Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
  Fact 1.0: Tom Gullikson was born on September 8, 1951.
  Fact 1.1: Tom Gullikson is a tennis coach.
  Fact 1.2: Tom Gullikson is a former professional tennis player.
  Fact 1.3: Tom Gullikson is from the United States.

Retrieved items:
<item 1>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 1>

<item 2>
Tom Gullikson
Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
</item 2>

<item 3>
2012 US Open – Men's Doubles
Jürgen Melzer and Philipp Petzschner were the defending champions, but lost in the second round to Jamie Delgado and Ken Skupski.

 Melzer and Petzschner needed a wildcard to compete because they forgot to sign up.

<br>
</item 3>

<item 4>
2010 Pilot Pen Tennis – Men's Doubles
Julian Knowle and Jürgen Melzer were the defending champions, but Melzer chose not to participate this year.As a result, Knowle partnered with Andy Ram, but
they lost to Robert Lindstedt and Horia Tecău in the semifinals.

 Lindstedt and Tecău went on to win the tournament, after defeating Rohan Bopanna and Aisam-ul-Haq Qureshi in the final 6–4, 7–5.
</item 4>

<item 5>
1983 Bristol Open – Doubles
Tim Gullikson and Tom Gullikson were the defending champions, but Tim Gullikson did not participate this year.

 Tom Gullikson partnered Johan Kriek, losing in the final.
</item 5>

<item 6>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 6>

<item 7>
Thomaz Koch
Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and
the U.S. National Championships.
</item 7>

<item 8>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 8>

<item 9>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 9>

<item 10>
Julian Knowle
Julian Knowle (born 29 April 1974) is an Austrian male professional tennis player.

 Being a born left-hander, Knowle is now one of the few on the ATP Tour who plays his forehand, backhand, and even volleys double-handed.

 He was Austria's most successful doubles player in history by reaching world no. 6 in the ATP doubles rankings in January 2008, before being matched by Jürgen
Melzer, who reached no. 6 in September 2010, and overtaken by Alexander Peya, who reached no. 3 in August 2013.
</item 10>
=== END ITEM id=086cb3c6e3a3bd2c ===

=== ITEM id=d20aad0cef7bbb6d ===
Gold sentences and facts:
Sentence 0: Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
  Fact 0.0: Javier Alberto Frana was born on 25 December 1966.
  Fact 0.1: Javier Alberto Frana is a former tennis player.
  Fact 0.2: Javier Alberto Frana is from Argentina.
  Fact 0.3: Javier Alberto Frana is a current tennis commentator.
  Fact 0.4: Javier Alberto Frana is a tennis commentator for ESPN Latin America.
Sentence 1: Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and the U.S. National Championships.
  Fact 1.0: Thomaz Koch was born on May 11, 1945.
  Fact 1.1: Thomaz Koch was born in Porto Alegre.
  Fact 1.2: Thomaz Koch is a left-handed former tennis player.
  Fact 1.3: Thomaz Koch is from Brazil.
  Fact 1.4: Thomaz Koch was a quarter-finalist at the French Open.
  Fact 1.5: Thomaz Koch was a quarter-finalist at Wimbledon.
  Fact 1.6: Thomaz Koch was a quarter-finalist at the U.S. National Championships.

Retrieved items:
<item 1>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 1>

<item 2>
Graph query row:
person: Thomaz Koch
occupation: former tennis player
</item 2>

<item 3>
1993 ATP Buenos Aires – Doubles
Pablo Albano and Javier Frana were the defending champions, but lost in the quarterfinals this year.
</item 3>

<item 4>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 4>

<item 5>
1996 IPB Czech Indoor – Doubles
Jonas Björkman and Javier Frana was the defending champions but did not compete that year.
</item 5>

<item 6>
1993 Movistar Open – Singles
Javier Frana defeated Emilio Sánchez 7–5, 3–6, 6–3 to win the 1993 Movistar Open singles competition.
</item 6>

<item 7>
Thomaz Koch
Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and
the U.S. National Championships.
</item 7>

<item 8>
Javier Frana
Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
</item 8>

<item 9>
1994 Movistar Open – Singles
Javier Frana was the defending champion but lost in the quarterfinals to Alberto Berasategui.
</item 9>

<item 10>
1995 Nottingham Open – Singles
Javier Frana defeated Todd Woodbridge 7–6, 6–3 in the final to secure the title.
</item 10>
=== END ITEM id=d20aad0cef7bbb6d ===
