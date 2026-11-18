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

=== ITEM id=b46cf4dedfb54003 ===
Gold sentences and facts:
Sentence 0:  Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.
  Fact 0.0: The club will compete in the league.
  Fact 0.1: The club will also participate in the FA Cup.
  Fact 0.2: The club will also participate in the EFL Cup.
  Fact 0.3: The club will also participate in the EFL Trophy.
Sentence 1: The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual knockout football competition in men's domestic English football.
  Fact 1.0: The EFL Cup is historically referred to as the League Cup.
  Fact 1.1: The EFL Cup is colloquially referred to as simply the League Cup.
  Fact 1.2: The EFL Cup is currently known as the Carabao Cup for sponsorship reasons.
  Fact 1.3: The EFL Cup is an annual knockout football competition.
  Fact 1.4: The EFL Cup is a competition in men's domestic English football.

Retrieved items:
<item 1>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 1>

<item 2>
EFL Cup
The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual
knockout football competition in men's domestic English football.

 Organised by the English Football League (EFL), it is open to any club within the top four levels of the English football league system – 92 clubs in total –
comprising the top level Premier League, and the three divisions of the English Football League's own league competition (Championship, League One and League
Two).
</item 2>

<item 3>
2017–18 Wigan Athletic F.C. season
The 2017–18 season is Wigan Athletic's 86th year in existence and their first back in League One, after being relegated the previous season.

 Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.

 Paul Cook was appointed as manager on 31 May 2017, signing a 3 year deal, following the departure of much of the previous coaching staff the day before.
</item 3>

<item 4>
Wigan Athletic returned to League One in 2017–18 under new manager Paul Cook, competing in four competitions. The club entered the Football League in 1978–79
and finished sixth; earlier seasons and its 2006–07 and 2007–08 Premier League campaigns are noted. Springfield Park hosted Wigan Athletic until 1999. Nick
Powell, an England youth international, rejoined Wigan in 2016. The material also covers Wigan Borough, Liverpool, Seattle
</item 4>

<item 5>
1979–80 Wigan Athletic F.C. season
The 1979–80 season was the 42nd season in the history of Wigan Athletic F.C. and their second as a professional club in the Football League.
</item 5>

<item 6>
2007–08 Wigan Athletic F.C. season
The 2007–08 Wigan Athletic F.C. season was the club's 30th season in the Football League and their third season in the Premier League.
</item 6>

<item 7>
1978–79 Wigan Athletic F.C. season
The 1978–79 season was the 41st season in the history of Wigan Athletic F.C. and their first as a professional club in the Football League.

 After finishing 2nd in the Northern Premier League during the previous season, the club was nominated to apply for Football League status, and were elected
into the league to replace Southport.

 After a poor start, the club exceeded expectations in the league, and towards the end of the season, a second consecutive promotion seemed possible.

 The club eventually fell short, finishing the season in 6th place with a total of 55 points.
</item 7>

<item 8>
2006–07 Wigan Athletic F.C. season
The 2006–07 Wigan Athletic F.C. season was the club's 29th season in the Football League and their second season in the Premier League.
</item 8>

<item 9>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 9>

<item 10>
List of Wigan Borough F.C. players
Wigan Borough F.C. was an English football club based in the town of Wigan.

 The club was founded in 1920 and joined the Lancashire Combination.

 In 1921, Borough turned professional when their application was accepted to play in the inaugural season of the newly formed Football League Third Division
North.

 The team played in the Football League for ten seasons, with their most successful season coming in 1928–29, finishing fourth in the league and reaching the
third round of the FA Cup.

 Wigan Borough folded during the 1931–32 season due to financial problems, and League football did not return to the town until Wigan Athletic F.C. were elected
into the Football League in 1978.
</item 10>
=== END ITEM id=b46cf4dedfb54003 ===

=== ITEM id=e8f41422dd561d3d ===
Gold sentences and facts:
Sentence 0: Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to become lead singer of the hard rock supergroup "Rock Star Supernova".
  Fact 0.0: Lukas Rossi was born on December 21, 1976.
  Fact 0.1: Lukas Rossi is a Canadian rock musician.
  Fact 0.2: Lukas Rossi was the winner of a CBS Television reality series.
  Fact 0.3: That CBS Television reality series was a televised audition contest to become lead singer of the hard rock supergroup 'Rock Star Supernova'.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath was born on January 31, 1946.
  Fact 1.1: Terry Alan Kath died on January 23, 1978.
  Fact 1.2: Terry Alan Kath was an American musician.
  Fact 1.3: Terry Alan Kath was an American songwriter.
  Fact 1.4: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.6: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Graph query row:
person: Lukas Rossi
nationality: Canadian
nationality_place: None
country: None
</item 1>

<item 2>
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
</item 2>

<item 3>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 3>

<item 4>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 4>

<item 5>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 5>

<item 6>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 6>

<item 7>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 7>

<item 8>
Graph query row:
person: Terry Alan Kath
nationality: American
nationality_place: None
country: None
</item 8>

<item 9>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 9>

<item 10>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 10>
=== END ITEM id=e8f41422dd561d3d ===

=== ITEM id=6706a67c6b082f24 ===
Gold sentences and facts:
Sentence 0: Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.
  Fact 0.0: Lake Louisvilla is a neighborhood.
  Fact 0.1: Lake Louisvilla is partially located in Louisville, Kentucky.
Sentence 1:  It is located between Westport Road in Louisville and KY 22 in Oldham County.
  Fact 1.0: Lake Louisvilla is located between Westport Road in Louisville.
  Fact 1.1: Lake Louisvilla is located between KY 22 in Oldham County.
Sentence 2: Oldham County is a county located in the commonwealth of Kentucky.
  Fact 2.0: Oldham County is a county.
  Fact 2.1: Oldham County is located in the commonwealth of Kentucky.
Sentence 3:  As of the 2010 census, the population was 60,316.
  Fact 3.0: As of the 2010 census, the population of Oldham County was 60,316.
Sentence 4:  Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.
  Fact 4.0: Oldham County was a prohibition or completely dry county until January 2005.
  Fact 4.1: Oldham County stopped being completely dry in January 2005 as the result of a 2004 'moist' vote.
  Fact 4.2: The 2004 'moist' vote permitted sales of alcohol in restaurants that seat at least 100 patrons.
  Fact 4.3: The 2004 'moist' vote permitted alcohol sales only in restaurants where 70% or more of total revenue is derived from sales of food.

Retrieved items:
<item 1>
John Logan (pioneer)
John Logan (1747July 1807) was a pioneer and politician from the U.S. state of Virginia and later, Kentucky.

 He participated in Lord Dunmore's War in 1774, serving under his brother, Benjamin.

 After moving to Kentucky County, Virginia, he took part in several expeditions against the Shawnee, including some led by Daniel Boone, John Bowman, and George
Rogers Clark.

 After Kentucky County was split into three counties, Logan represented his home county, Lincoln in the Virginia House of Delegates and at several of the
conventions that effected the separation of Virginia from Kentucky.
</item 1>

<item 2>
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 2>

<item 3>
Garrard County, Kentucky
Garrard County ( ;) is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 16,912.

 Its county seat is Lancaster.

 The county was formed in 1796 and was named for James Garrard, Governor of Kentucky from 1796 to 1804.

 It is a prohibition or dry county but Lancaster is wet.

 Lancaster was founded as a settlement of log cabins in 1776 at a springs that later provided a constant source of water to early pioneers.

 It is one of the oldest cities in the Commonwealth.

 Boonesborough, 25 miles to the east, was founded by Daniel Boone in 1775.

 Lexington, 28 miles to the north, was founded in 1775.

 Stanford, originally known as St. Asaph, is 10 miles south of Lancaster.

 It too was founded in 1775.

 The oldest permanent settlemenet in Kentucky, Harrodsburg, was founded in 1774 and is 18 miles to the west.

 The present day courthouse is one of the oldest courthouses in Kentucky in continuous use.
</item 3>

<item 4>
Casey County, Kentucky
Casey County is a county located in the U.S. Commonwealth of Kentucky.

 As of the 2010 census, the population was 15,955.

 Its county seat is Liberty.

 The county was formed in 1806 from the western part of Lincoln County and named for Colonel William Casey, a pioneer settler who moved his family to Kentucky
in 1779.

 It is the only Kentucky county entirely in the Knobs region.

 Casey County is home to annual Casey County Apple Festival, and is a prohibition or dry county.

 It is considered part of the Appalachian region of Kentucky.
</item 4>

<item 5>
Kentucky County, Virginia
Kentucky County (then alternately spelled Kentucke County) was formed by the Commonwealth of Virginia by dividing Fincastle County into three new counties:
Kentucky, Washington, and Montgomery, effective December 31, 1776.

 During the three and one-half years of Kentucky County's existence, its seat of government was Harrodstown (then also known as Oldtown, later renamed
Harrodsburg).
</item 5>

<item 6>
Oldham County, Kentucky
Oldham County is a county located in the commonwealth of Kentucky.

 As of the 2010 census, the population was 60,316.

 Its county seat is La Grange.

 The county is named for Colonel William Oldham.

 Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants
that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.

 After a vote in late 2015; Oldham county has become a completely wet county.
</item 6>

<item 7>
Lake Louisvilla, Louisville
Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.

 It is located between Westport Road in Louisville and KY 22 in Oldham County.

 Lake Louisvilla was developed in the 1920s as a summer resort for people living in the city of Louisville.

 The state of Kentucky drained the lake in the late 1980s due to safety concerns regarding the stability of a dam.
</item 7>

<item 8>
Wheeling Island
Wheeling Island is the most populated island in the Ohio River.

 It lies within the city of Wheeling in Ohio County, West Virginia, in the United States.

 The 2000 census showed a resident population of 3,142 people on the island, which has a land area of 1.514 km² (374 acres).

 Neville Island, PA is larger (3.4 km²) but has fewer people (1,232).
</item 8>

<item 9>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 9>

<item 10>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 10>
=== END ITEM id=6706a67c6b082f24 ===

=== ITEM id=c2f30c41019a7496 ===
Gold sentences and facts:
Sentence 0:  One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
  Fact 0.0: Updike was one of only three writers to win the Pulitzer Prize for Fiction more than once.
  Fact 0.1: The other two writers who won the Pulitzer Prize for Fiction more than once were Booth Tarkington and William Faulkner.
  Fact 0.2: Updike published more than twenty novels during his career.
  Fact 0.3: Updike published more than a dozen short-story collections during his career.
  Fact 0.4: Updike published poetry during his career.
  Fact 0.5: Updike published art and literary criticism during his career.
  Fact 0.6: Updike published children's books during his career.
Sentence 1:  Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.
  Fact 1.0: Seventeen of Updike's novels were bestsellers.
  Fact 1.1: More than 100 million copies of Updike's books are in print.

Retrieved items:
<item 1>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 1>

<item 2>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 2>

<item 3>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 3>

<item 4>
Tom Clancy's Net Force Explorers: High Wire
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 4>

<item 5>
*The 39 Clues: Doublecross*, the series’ fourth installment, follows a divided Cahill family. Ian Kabra’s leadership is challenged by a coup led by the Outcast,
who forces the family to avert historical disasters; revelations include Grace’s cruelty and the Outcast’s identity as Nathaniel Hartford.

*Net Force Explorers* is a Tom Clancy–Steve Pieczenik young-adult spin-off; titles listed include *Deathworld*, *Safe House*,
</item 5>

<item 6>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 6>

<item 7>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 7>

<item 8>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 8>

<item 9>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 9>

<item 10>
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 10>
=== END ITEM id=c2f30c41019a7496 ===

=== ITEM id=feebff819e4c82dd ===
Gold sentences and facts:
Sentence 0:  Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any affiliation with the work.
  Fact 0.0: The film takes its title from an album by Leonard Cohen.
  Fact 0.1: Ira Nadel wrote a biography of Leonard Cohen.
  Fact 0.2: The subject of the film is not Leonard Cohen.
  Fact 0.3: The subject of the film has no affiliation with the work.
Sentence 1: Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and painter.
  Fact 1.0: Leonard Norman Cohen was born on September 21, 1934.
  Fact 1.1: Leonard Norman Cohen died on November 7, 2016.
  Fact 1.2: Leonard Norman Cohen was a Canadian singer.
  Fact 1.3: Leonard Norman Cohen was a Canadian songwriter.
  Fact 1.4: Leonard Norman Cohen was a Canadian musician.
  Fact 1.5: Leonard Norman Cohen was a Canadian poet.
  Fact 1.6: Leonard Norman Cohen was a Canadian novelist.
  Fact 1.7: Leonard Norman Cohen was a Canadian painter.

Retrieved items:
<item 1>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 1>

<item 2>
For Those Who Think Young (album)
For Those Who Think Young, appearing on the album cover as (for those who think young) and originally to be entitled for those who think jung, was the third
album by Canadian new wave band Rough Trade; it was released in 1981 (True North TN-48 in Canada; Boardwalk NB-33261-1, US; Big Time, UK; CBS 85385, The
Netherlands).

 It climbed to #9 in Canada on the "RPM" Top 50 Albums Chart on November 7, 1981 (putting it at #1 on the CANCON Chart listing), and held the position for three
weeks, dropping out of the Top 50 after sixteen weeks on February 6 of the following year.

 It was certified gold in Canada by the CRIA on November 1, 1981.

 The single "All Touch" gave the band its biggest commercial success, reaching #12 in Canada on the "RPM" Top 50 Singles Chart (#2 on the CANCON Chart) and #58
on the U.S. ""Billboard" Hot 100".
</item 2>

<item 3>
Various Positions Tour
The Various Positions Tour was a concert tour by Leonard Cohen, in support of his album "Various Positions", released in 1984 (Canada) and 1985 in Europe and
the United States.
</item 3>

<item 4>
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 4>

<item 5>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 5>

<item 6>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 6>

<item 7>
Jack Johnson (album)
Jack Johnson, later reissued as A Tribute to Jack Johnson, is a 1971 studio album and soundtrack by American jazz trumpeter, composer, and bandleader Miles
Davis.

 In 1970, Davis was asked by Bill Cayton to record music for his documentary of the same name on the life of boxer Jack Johnson.

 Johnson's saga resonated personally with Davis, who wrote in the album's liner notes of Johnson's mastery as a boxer, his affinity for fast cars, jazz,
clothes, and beautiful women, his unreconstructed blackness, and his threatening image to white men.

 This was the second film score he had composed, after "Ascenseur pour l'échafaud" in 1957.
</item 7>

<item 8>
Daddy's Hands (band)
Daddy's Hands was a Canadian experimental rock band formed in Victoria, British Columbia, Canada, by singer/guitarist Dave Wenger, bassist Emily Bauslaugh,
drummer Jonah Fortune, and saxophonist Jonathan Pollard.

 Dave Wenger and Emily Bauslaugh had been involved in the British Columbian hardcore/emo scene before forming the band, being in bands M Blanket and Ache Hour
Credo, and the band Floragore respectively.

 Daddy's Hands has released two demo tapes, one EP and two full-length albums.

 The band gigged extensively in the 90s, even playing a radio show with Nardwuar.

 Despite achieving little commercial success, the band has been referred to as highly influential in the West Coast music scene.

 Lead singer Wenger, highly regarded by some critics, has been referred to as Montreal's Peter Laughner.
</item 8>

<item 9>
Love Can't Wait (Nick Carter song)
"Love Can't Wait" is a song by American singer-songwriter Nick Carter, released as the lead Canadian single from his second studio album, "I'm Taking Off".

 The single was released on June 21, 2011.

 The song only appears on the Canadian version of the album, and has not been released in any other territories.

 The music video for the song features Carter performing the song with the backdrop of a motorway system.

 The music video also featured Canadian pop star Shawn Desman, who co-wrote the song.

 The single peaked at #2 on the Canadian Singles Chart, based entirely on download sales.

 Carter ran a competition for amateur mixers to remix the track, with the winning remixer earning $1500 and their remix appearing on Carter's remix album, "I'm
Taking Off: Relaunched and Remixed".
</item 9>

<item 10>
The Cold Nose
The Cold Nose is the debut album by Department of Eagles.

 It has been reissued twice.

 It was originally released in the US as The Whitey On The Moon UK LP on Isota Records on October 21, 2003, and reissued in the UK as The Cold Nose on Melodic
Records on January 8, 2008.

 The American Dust October 7, 2008 US reissue contains four bonus tracks.

 The album is sample-heavy and multilayered.

 The second track, "Sailing By Night," contains a sample of the song "Meetings Along The Edge" from the Philip Glass and Ravi Shankar project "Passages", as
well as a sample from the Yellow Magic Orchestra track, Rydeen.

 The third track "Noam Chomsky Spring Break 2002" samples the Regina Spektor song "Prisoners".

 It also samples Astor Piazzolla's composition "Oblivion" in the opening and closing sequences.
</item 10>
=== END ITEM id=feebff819e4c82dd ===

=== ITEM id=cda3428d3bea60fe ===
Gold sentences and facts:
Sentence 0:  The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.
  Fact 0.0: The pool lies directly south of Lichfield Cathedral.
  Fact 0.1: The pool has historically been important to the defence of the Cathedral Close.
Sentence 1:  The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.
  Fact 1.0: The Diocese of Lichfield covers all of Staffordshire.
  Fact 1.1: The Diocese of Lichfield covers much of Shropshire.
  Fact 1.2: The Diocese of Lichfield covers part of the Black Country.
  Fact 1.3: The Diocese of Lichfield covers part of the West Midlands.

Retrieved items:
<item 1>
Minster Pool
Minster Pool is a reservoir located between Bird Street and Dam Street in the heart of the city of Lichfield, Staffordshire in the United Kingdom.

 The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.

 The pool was originally formed in the 11th century when a boggy stream was dammed at its eastern end to drive a mill on Dam Street.

 The pool was used as a mill pond and fishery until 1856 when the mill was demolished; it has since been retained for public amenity.
</item 1>

<item 2>
Walsall
Walsall ( ) is an industrial town in the West Midlands of England.

 It is located 8 miles north-west of the City of Birmingham and 6 miles east of the City of Wolverhampton.

 Historically a part of Staffordshire, Walsall is a component area of the West Midlands conurbation, and part of the Black Country.
</item 2>

<item 3>
Dudley Zoo
Dudley Zoological Gardens is a 40 acre zoo located within the grounds of Dudley Castle in the town of Dudley, in the Black Country region of the West Midlands,
England.

 The Zoo opened to the public on 18 May 1937.

 Dudley Zoo is owned and operated by Dudley and West Midlands Zoological Society, founded in 1935 and a registered charity.
</item 3>

<item 4>
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
</item 4>

<item 5>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 5>

<item 6>
Brierley Hill
Brierley Hill is a small town and electoral ward of the Dudley Metropolitan Borough, in the West Midlands of England, and is situated approximately 2.5 miles
south of central Dudley and 2 miles north of Stourbridge.

 Part of the Black Country, and in a heavily industrialised area of the Dudley Borough, it has a population of 13,935 at the 2011 census, and is best known for
glass and steel manufacturing, although the industry has declined considerably since the 1970s.

 One of the largest factories in the area was the Round Oak Steelworks, which was closed down and redeveloped in the 1980s to become the Merry Hill Shopping
Centre.

 Brierley Hill was originally in Staffordshire, but is now part of the West Midlands metropolitan county since its creation in 1974.
</item 6>

<item 7>
Tipton
Tipton is a town in the borough of Sandwell, West Midlands, England, with a population of around 38,777 at the 2011 UK Census.

 Tipton is located about halfway between Birmingham and Wolverhampton.

 It is a part of the West Midlands conurbation and is a part of the Black Country.
</item 7>

<item 8>
Black Country Rangers F.C.
Black Country Rangers Football Club is a football club based in Rowley Regis in the West Midlands.

 They are currently members of the West Midlands (Regional) League Premier Division and play at the Beeches in Cradley, groundsharing with Cradley Town.
</item 8>

<item 9>
List of railway stations in the West Midlands
This is a list of railway stations within the West Midlands, a metropolitan county in central England which includes the cities of Birmingham, Coventry and
Wolverhampton.

 It includes all railway stations in the West Midlands that currently have regular timetabled train services, as well as certain stations outside the county
which are within the area supported by Transport for West Midlands (TfWM), formerly known as Centro.

 Transport within the West Midlands is subsidised by TfWM, who since 2006 have used the brand name Network West Midlands to demonstrate the 'joined-up' nature
of the regions bus and rail networks.
</item 9>

<item 10>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 10>
=== END ITEM id=cda3428d3bea60fe ===

=== ITEM id=9c507971381f52da ===
Gold sentences and facts:
Sentence 0:  Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the summer.
  Fact 0.0: Obertraun is a popular holiday destination.
  Fact 0.1: Obertraun offers skiing and snowboarding in the winter.
  Fact 0.2: Obertraun offers mountain biking in the summer.
  Fact 0.3: Obertraun offers swimming and kayaking in the summer.
Sentence 1:  The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the precious salt mines of the Habsburg Monarchy.
  Fact 1.0: The name 'Salzkammergut' literally means 'Estate of the Salt Chamber'.
  Fact 1.1: The name 'Salzkammergut' derives from the Imperial Salt Chamber.
  Fact 1.2: The Imperial Salt Chamber was the authority charged with running the precious salt mines of the Habsburg Monarchy.

Retrieved items:
<item 1>
Obertraun
Obertraun, Upper Austria is a village in the Salzkammergut, a region in Austria.

 It is located near the Hallstätter See (Hallstatt Lake) and Hoher Dachstein.

 Obertraun is a popular holiday destination offering activities such as skiing, snowboarding in the winter and mountain biking, swimming and kayaking in the
summer.
</item 1>

<item 2>
Hundsheimer Berg
Hundsheimer Berg is a hill located in the Hundsheimer Berge hill range in Lower Austria, Austria close to the border with Slovakia.

 Its peak is 481 m above sea level which makes it the highest hill in the Hundsheimer Berge.

 The north slopes of the Hundsheimer Berg are covered with forests while the southern slopes have a steppe vegetation.

 Since 1965, a nature reserve with the same name, "Hundsheimer Berg", is located on these southern slopes.
</item 2>

<item 3>
South Bergish
South Bergish (German: "Südbergische Dialekte ") or Upper Bergish (German: "Oberbergische Dialekte ") is a group of German dialects of the Bergisches Land
Region East of the Rhine and approximately south of the Wupper and north of the Sieg.

 These dialects are part of the Ripuarian group and thus are also called East Ripuarian.

 Ripuarian dialects are also spoken west of the Rhine up to the German border, and in some small areas next to the respective borders in Belgium and in the
Netherlands.

 Ripuarian Bergish dialects belong to the Middle German group, and thus are varieties of High German, where they belong to the northmost ones.

 In the North, they border to the East Bergish and the West Bergish aka Low Bergish language groups, which are part of the Low Franconian group like Dutch.
</item 3>

<item 4>
The Inheritors (1998 film)
The Inheritors (original German title "Die Siebtelbauern" – "The Seventh-Part Farmers") is a 1998 Austrian-German film directed by Stefan Ruzowitzky.

 It stars Simon Schwarz and Sophie Rois and has won numerous awards.

 The film was selected as the Austrian entry for the Best Foreign Language Film at the 71st Academy Awards, but was not accepted as a nominee.
</item 4>

<item 5>
Upper Saxon German
Upper Saxon (German: "Obersächsisch" ) is an East Central German dialect spoken in much of the modern German State of Saxony and in the adjacent parts of
Saxony-Anhalt and Thuringia.

 Though colloquially called "Saxon" ("Sächsisch" ), it is not to be confused with the Low Saxon dialect group in Northern Germany.

 Upper Saxon is closely linked to the Thuringian dialect spoken in the adjacent areas to the west.
</item 5>

<item 6>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 6>

<item 7>
List of Austrian mountain climbers
This is a list of Austrian mountaineers.

 On 5 of the 14 Eight-thousanders Austrians have made the first ascent, more than any other nation can claim.

 Out of the total of 9 Austrian mountaineers who made first ascents of Eight-thousanders, 3 have been members of the Edelweiss Club Salzburg, an association of
mountaineers founded in Salzburg in 1881
</item 7>

<item 8>
Appenzell Alps
The Appenzell Alps (German: "Appenzeller Alpen" ) are a mountain range in Switzerland on the northern edge of the Alps.

 They extend into the cantons of Appenzell Ausserrhoden, Appenzell Innerrhoden and St. Gallen and are bordered by the Glarus Alps to the west and the Rätikon to
the south-east.
</item 8>

<item 9>
Salzkammergut
The Salzkammergut is a resort area located in Austria.

 It stretches from the City of Salzburg eastwards along the Austrian Alpine Foreland and the Northern Limestone Alps to the peaks of the Dachstein Mountains,
spanning the federal states of Upper Austria, Salzburg, and Styria.

 The main river of the region is the Traun, a right tributary of the Danube.

 The name "Salzkammergut" literally means "Estate of the Salt Chamber" and derives from the Imperial Salt Chamber, the authority charged with running the
precious salt mines of the Habsburg Monarchy.

 It is a UNESCO World Heritage Site.
</item 9>

<item 10>
Crystal Mountain (British Columbia)
Crystal Mountain Resort was a small day-use ski area near West Kelowna, British Columbia, Canada.

 It has two chairlifts and one surface lift: a GMD Mueller doublechair, a Leitner-Poma triplechair, and a Doppelmayr T-bar.

 The ski area has 30 designated groomed runs and the resort also has many different tree trails and some glades.

 Despite the smaller size of the resort, which is considerably smaller than neighbouring resorts such as Big White Ski Resort and Silver Star Mountain Resort,
the mountain is of good size and is a popular destination for both experienced and first-time skiers, snowboarders and snowshoers.

 Despite the lower elevation, the resort receives heavy snowfall each year with many powder days throughout the winter season.

 It was long known as Last Mountain Ski Resort but has since changed its name to Crystal Mountain Resort in 1992.

 Crystal Mountain Resort offers lessons for skiing and snowboarding as well as offering rentals for snowshoeing.

 It has not been in operation since 2014 due to a lift malfunction of the Blue doublechair.
</item 10>
=== END ITEM id=9c507971381f52da ===

=== ITEM id=1dc7e31f7b597378 ===
Gold sentences and facts:
Sentence 0:  The Park is home to the first flume ride in Ireland (Pirates Plunge).
  Fact 0.0: The Park is home to the first flume ride in Ireland.
  Fact 0.1: The first flume ride in Ireland is named Pirates Plunge.
Sentence 1: Grace O'Malley (c. 1530 – c. 1603; also Gráinne O'Malley, Irish: "Gráinne Ní Mháille" ) was lord of the Ó Máille dynasty in the west of Ireland, following in the footsteps of her father Eoghan Dubhdara Ó Máille.
  Fact 1.0: Grace O'Malley lived from about 1530 to about 1603.
  Fact 1.1: Grace O'Malley is also called Gráinne O'Malley.
  Fact 1.2: Grace O'Malley's Irish name is Gráinne Ní Mháille.
  Fact 1.3: Grace O'Malley was lord of the Ó Máille dynasty in the west of Ireland.
  Fact 1.4: Grace O'Malley followed in the footsteps of her father, Eoghan Dubhdara Ó Máille.

Retrieved items:
<item 1>
Grizzly River Run
Grizzly River Run is located at Disney California Adventure at the Disneyland Resort in Anaheim, California.

 It is similar to Kali River Rapids in Disney's Animal Kingdom but distinctive as the rafts are engineered to spin as they descend chutes.

 The attraction's name comes from Grizzly Peak, the bear shaped mountain that the rapids flow around.

 It was designed by Walt Disney Imagineering and constructed by Intamin.
</item 1>

<item 2>
Glenwood Caverns Adventure Park
Glenwood Caverns Adventure Park is an adventure park located above Glenwood Springs, Colorado, about 160 miles west of Denver.

 Prior to 2003, only cave tours were available until a major expansion took place.

 The park is unique because it sits at an altitude of 7,100 feet on a mountain above Glenwood.

 Today, the park features numerous attractions in addition to the cave tours.
</item 2>

<item 3>
Luna Park, Schenectady
Roughly seven decades before the Skycoaster rides that now dot various United States amusement parks, Luna/Rexford Park featured an aerial swing ride.
</item 3>

<item 4>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 4>

<item 5>
Tuckaleechee Caverns
Estimated to be between 20 and 30 million years old, the Caverns are rich in history and lore in recent years as well.

.
</item 5>

<item 6>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 6>

<item 7>
Treasure Island Resort &amp; Casino
In 1993, a 78,000-square-foot expansion was added which created three new restaurants, valet parking, state-of-the-art kitchen, a gift shop, players club,
ballroom and a new entertainment area.

 In the following year, a 137-slip marina and 95-site RV park would open.

 Growth continued as 9,854-square-foot addition for business offices is established in 1995.

 In 1996, Treasure Island made a big step with a $20 million addition and redesign.

 A strategic marketing shift changed the name to Treasure Island Resort & Casino with the addition of new theming and a 250-room hotel transformed Treasure
Island into a destination resort.

 The total square footage has 350,000 with 25,000 square feet designed for meeting space.

 In 2001, an additional 200,000 feet were added to the casino, which included a new great entry, higher ceilings to improve air quality, additional games,
70,000-square-foot office space and 60,000-square-foot warehouse.
</item 7>

<item 8>
White Water Landing (Dorney Park)
White Water Landing is a Shoot-the-Chutes ride at Dorney Park & Wildwater Kingdom amusement park in Allentown, Pennsylvania.

 It was built in 1992, 10 years after a sister park named Cedar Point Built a defunct flume using the same name: White Water Landing (Cedar Point).

 This ride is identical to Snake River Falls at Cedar Point in Sandusky, Ohio.

 It is located in the upper side near Hydra the Revenge.
</item 8>

<item 9>
Christchurch Adventure Park
Christchurch Adventure Park is an Adventure park in the Port Hills of Christchurch, New Zealand.

 Built by the Canadian company Select Evolution, it had its opening function on 16 December 2016.

 The 1500 holders of special passes could ride from 17 December, while the park opened to the public on 21 December.

 According to the developer, the 358 ha park with a 1.8 km chairlift and initially 50 km of downhill tracks is the largest facility of its kind in the Southern
Hemisphere.

 The park closed on 13 February as a precaution due to a nearby fire and two days later, most of the park's tree cover had been destroyed in a large wildfire.

 The lift and ziplines have suffered significant damage, and the park's condition is much worse than the operator initially thought.

 No reopening date has been set, and "it won't be a quick fix".
</item 9>

<item 10>
Pirate Adventure Park
Pirate Adventure Park is a small theme park located beside Westport House in Westport, Ireland.

 The Park is home to the first flume ride in Ireland (Pirates Plunge).

 The Park is Pirate themed due to Wesport House's association with Grace O'Malley.

 The Park was developed by The 11th Marquess of Sligo on the grounds of Westport House.
</item 10>
=== END ITEM id=1dc7e31f7b597378 ===

=== ITEM id=32d56c33ba1bca0b ===
Gold sentences and facts:
Sentence 0:  Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
  Fact 0.0: Cacareco's candidacy inspired the Rhinoceros Party of Canada.
  Fact 0.1: The Rhinoceros Party of Canada was nominally led by the rhinoceros Cornelius the First.
Sentence 1: The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.
  Fact 1.0: The Rhinoceros Party's French name is 'Parti Rhinoceros'.
  Fact 1.1: The Rhinoceros Party was a registered political party in Canada.
  Fact 1.2: The Rhinoceros Party was a registered political party in Canada from the 1960s to the 1990s.

Retrieved items:
<item 1>
Civil Human Rights Front
Civil Human Rights Front or CHRF () is an organisation that focuses on the issues of Hong Kong politics and livelihood, affiliates almost all the pan-democratic
camps in Hong Kong.

 Forty-eight NGOs and political groups have been involved in the organisation as of January 2006.

 The most well-known event held by the CHRF is the Hong Kong 1 July marches.
</item 1>

<item 2>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 2>

<item 3>
Revolutionary Communist Party of Canada
The Revolutionary Communist Party of Canada, or RCP Canada, is a revolutionary communist party oriented around Marxism-Leninism-Maoism.

 The creation of the organizational stage of the Party was adopted at what was called the "Revolutionary Communist Conference", which was held in Montreal,
Quebec in November 2000 by activists and former members of the labour union movements and youth organizations of the 1960s, 1970s and 1980s, who felt that the
revolutionary situation in Canada warranted the creation of a Party dedicated to a communist revolution.

 The party was initially called the Revolutionary Communist Party (Organizing Committees) (RCP(OC)).

 At this conference, participants adopted the Party's first Draft Programme.

 Because of the location of the conference, the majority of the founding members were French-speaking Quebecers, and the Party began an extensive effort to
reach out to the rest of Canada, starting with the Canadian Revolutionary Congress held in November 2006 in Toronto.

 It is not recognized as a political party by Elections Canada, because the party rejects what it calls the "bourgeois electoral system" and doesn't seek
recognition by the State.
</item 3>

<item 4>
LPRadicals
LPRadicals is a caucus formed in 2006 within the United States Libertarian Party by Susan Hogarth and other party members who opposed removal of much of the
material in the party platform during the 2006 national party convention.

 The caucus lists four points as "key strategic principles in furthering the work of the Libertarian Party toward, as our platform describes it, 'a world set
free in our lifetime'."

 These are: Rights Are Utilitarian, Radical Abolitionism, Principled Populism, and No Particular Order (for removal of government policies).

 The caucus was active at the 2008 and 2010 Libertarian National Conventions.

 The radical caucus was revived and was extraordinarily active during the 2016 Libertarian National Convention.
</item 4>

<item 5>
History of the United States Republican Party
The Republican Party, also commonly called the GOP (for "Grand Old Party"), is one of the world's oldest extant political parties.

 It is the second oldest existing political party in the United States after its primary rival, the Democratic Party.

 It emerged in 1854 to combat the Kansas–Nebraska Act, an act that dissolved the terms of the Missouri Compromise and allowed slave or free status to be decided
in the territories by popular sovereignty.

 The Party had almost no presence in the Southern United States, but by 1858 in the North it had enlisted former Whigs and former Free Soil Democrats to form
majorities in nearly every Northern state.
</item 5>

<item 6>
Cacareco
Cacareco (1954-1962) was a female black rhinoceros exhibited in Brazilian zoos.

 She became famous as a candidate for the 1958 São Paulo city council elections with the intention of protesting against political corruption.

 Electoral officials did not accept Cacareco's candidacy, but she eventually won 100,000 votes, more than any other party in that same election (which was also
marked by rampant absenteeism).

 Today, the term "voto Cacareco" (Cacareco vote) is commonly used to describe protest votes in Brazil.

 Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
</item 6>

<item 7>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 7>

<item 8>
Rhinoceros Party of Canada (1963–93)
The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.

 Operating within the tradition of political satire, the Rhinoceros Party's basic credo, their so-called primal promise, was "a promise to keep none of our
promises".

 They then promised outlandishly impossible schemes designed to amuse and entertain the voting public.
</item 8>

<item 9>
Aboriginal Peoples Party of Canada
The Aboriginal Peoples Party of Canada (APP) is a Canadian political party that was founded in 2005.

 The party was conceived by University of Lethbridge student Myron Wolf Child.

 It held its founding meeting on August 21, 2005, in St. Albert, Alberta.

 The APP was headed by interim leader Bill Montour, a former Chief of the Six Nations Reserve in Ontario.

 In October 2005, the party had 122 confirmed members, just less than half the number needed to register as an official party in Canada.

 In late October 2005, the APP sought to unite with the First Peoples National Party of Canada which also had fewer than the number of confirmed members needed
to become a registered political party.

 The First Peoples National Party of Canada became an eligible political party on December 6, 2005.

 Whether or not this was accomplished through a merger with the APP is unclear.

 If this is the case, the APP no longer exists, but, as one of the stipulations of a possible merger was that a name for the new party would be determined at
the party's first convention, it is possible that the name Aboriginal Peoples Party of Canada will come into usage again.
</item 9>

<item 10>
National Advancement Party of Canada
The National Advancement Party of Canada (French: "Parti de l'Essor National du Canada" ) is a minor federal political party in Canada.

 The party describes itself as for "advancement of the Canadian people and Canada".

 The party was formed in 2014 and registered with Elections Canada in 2015, initially under the name Democratic Advancement Party of Canada.

 The party is headquartered in Calgary, Alberta and led by writer Stephen Garvey.
</item 10>
=== END ITEM id=32d56c33ba1bca0b ===

=== ITEM id=98cf692eca2c13e6 ===
Gold sentences and facts:
Sentence 0:  Jan Kodeš and Margaret Court won the singles titles.
  Fact 0.0: Jan Kodeš won the singles titles.
  Fact 0.1: Margaret Court won the singles titles.
Sentence 1:  She is currently a Christian minister in Perth, Western Australia.
  Fact 1.0: A woman referred to in this sentence is currently a Christian minister.
  Fact 1.1: The Christian minister referred to in this sentence is based in Perth, Western Australia.

Retrieved items:
<item 1>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 1>

<item 2>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 2>

<item 3>
1972 French Open – Men's Singles
Jan Kodes was the two-time defending champion, but lost in the quarterfinals to Patrick Proisy.
</item 3>

<item 4>
1970 French Open
The 1970 French Open was a tennis tournament that was held at the Stade Roland Garros in Paris in France from 25 May through 7 June 1970.

 It was the 69th edition of the French Open, the 40th to be open to foreign competitors, and the second Grand Slam of the year.

 Jan Kodeš and Margaret Court won the singles titles.
</item 4>

<item 5>
Margaret Court
Margaret Court {'1': ", '2': ", '3': ", '4': "} (née Smith; born 16 July 1942), also known as Margaret Smith Court, is a retired Australian tennis player and
former world No. 1.

 She is currently a Christian minister in Perth, Western Australia.

 In tennis, she amassed more major titles than any other player in history.
</item 5>

<item 6>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 6>

<item 7>
Graph query row:
person: Margaret Court
relationship: AWARDED
connected_entity: singles titles
residence_city: Perth
</item 7>

<item 8>
Carlos Fernandes (tennis)
Carlos Fernandes (1936-) is a retired Brazilian tennis player.

 He had a good all-round game with excellent passing shots and moved fast around the court.

 Fernandes had a reputation for being a charismatic ladies man.

 He later became a coach.

 He made his Grand Slam debut at Wimbledon 1957, losing in the opening round to Jorgen Ulrich.

 At Roland Garros in 1958, Fernandes lost in round one to Pierre Darmon.

 At Wimbledon he lost in round one to Butch Buchholz.

 At 1959 French Open, Fernandes lost in round three to Jacques Brichant.At Wimbledon he lost in round two to Neale Fraser.

 At French Open 1960, Fernandes lost in round three to Bobby Wilson and at Wimbledon lost in round two to Wolfgang Stuck.

 At 1961 French Open, Fernandes achieved his best Grand Slam singles result by beating Pierre Darmon and Bob Hewitt before losing to Jan-Erik Lundquist in the
quarter finals.

 He lost in round two of Wimbledon to Wilson.

 At Roland Garros 1962, Fernandes lost to Lundquist in round two.
</item 8>

<item 9>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 9>

<item 10>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 10>
=== END ITEM id=98cf692eca2c13e6 ===

=== ITEM id=d0412924c195efe9 ===
Gold sentences and facts:
Sentence 0: Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.
  Fact 0.0: Rosario Castellanos Figueroa was born on 25 May 1925.
  Fact 0.1: Rosario Castellanos Figueroa died on 7 August 1974.
  Fact 0.2: Rosario Castellanos Figueroa was a Mexican poet.
  Fact 0.3: Rosario Castellanos Figueroa was a Mexican author.
Sentence 1: Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a series of novels beginning in 1908 with "Anne of Green Gables".
  Fact 1.0: Lucy Maud Montgomery was born on November 30, 1874.
  Fact 1.1: Lucy Maud Montgomery died on April 24, 1942.
  Fact 1.2: Lucy Maud Montgomery published as L.M. Montgomery.
  Fact 1.3: Lucy Maud Montgomery was a Canadian author.
  Fact 1.4: Lucy Maud Montgomery is best known for a series of novels.
  Fact 1.5: The series of novels began in 1908 with Anne of Green Gables.

Retrieved items:
<item 1>
Mollie Gillen
Mollie Gillen (née Woolnough; 1908–2009) was an Australian historian, researcher, writer and novelist.

 Her work on the First Fleet, in "The search for John Small," "First Fleeter" and The Founders of Australia: A Biographical Dictionary of the First Fleet,
explored the idea that many of the founding families of Australia were descended from the convict population, rather than those sent to guard them.

 Gillen's article Maud Montgomery: The Girl Who Wrote Green Gables instigated a new era in scholarship on Lucy Maud Montgomery.
</item 1>

<item 2>
Emily's Quest
Emily's Quest is a novel and the last of the "Emily" trilogy by Lucy Maud Montgomery.

 After finishing "Emily Climbs", Montgomery suspended writing "Emily's Quest" and published "The Blue Castle"; she resumed writing and published in 1927.
</item 2>

<item 3>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 3>

<item 4>
Anne Shirley
Anne Shirley is a fictional character introduced in the 1908 novel "Anne of Green Gables" by Lucy Maud Montgomery.

 Montgomery wrote in her journal that the idea for Anne's story came from relatives who, planning to adopt an orphaned boy, received a girl instead.

 Anne Shirley's appearance was inspired by a photograph which Montgomery clipped from the Metropolitan Magazine and kept, unaware of the model's identity as the
1900s Gibson Girl Evelyn Nesbit.
</item 4>

<item 5>
L.M. Montgomery's Anne of Green Gables
L.M. Montgomery's Anne of Green Gables is a Canadian television film based on Lucy Maud Montgomery's novel of the same name.

 It first aired on YTV on February 15, 2016 and starred Ella Ballentine, Martin Sheen and Sara Botsford.

 Montgomery's granddaughter, Kate Macdonald Butler, was one of the film's executive producers.

 The film's world premiere was held February 2, 2016 at the Canadian Museum of History.
</item 5>

<item 6>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 6>

<item 7>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 7>

<item 8>
Anne of Ingleside
Anne of Ingleside is a children's novel by Canadian author Lucy Maud Montgomery.

 It was first published in July 1939 by McClelland and Stewart (Toronto) and the Frederick A. Stokes Company (New York).

 It is the tenth of eleven books that feature the character of Anne Shirley, and Montgomery's final published novel.

 (Two novels that occur later in the "Anne" chronology were actually published years earlier.

 As well, the short story collection "The Blythes Are Quoted", written in 1941/42, but not published until 2009, concludes the Anne chronology.)
</item 8>

<item 9>
Anne of Green Gables
Anne of Green Gables is a 1908 novel by Canadian author Lucy Maud Montgomery (published as L. M. Montgomery).

 Written for all ages, it has been considered a children's novel since the mid-twentieth century.

 It recounts the adventures of Anne Shirley, an 11-year-old orphan girl who is mistakenly sent to Matthew and Marilla Cuthbert, a middle-aged brother and sister
who had intended to adopt a boy to help them on their farm in the fictional town of Avonlea on Prince Edward Island.

 The novel recounts how Anne makes her way with the Cuthberts, in school, and within the town.
</item 9>

<item 10>
Rosario Castellanos
Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.

 Along with the other members of the Generation of 1950 (the poets who wrote following the Second World War, influenced by César Vallejo and others), she was
one of Mexico's most important literary voices in the last century.

 Throughout her life, she wrote eloquently about issues of cultural and gender oppression, and her work has influenced feminist theory and cultural studies.

 Though she died young, she opened the door of Mexican literature to women, and left a legacy that still resonates today.
</item 10>
=== END ITEM id=d0412924c195efe9 ===
