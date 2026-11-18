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

=== ITEM id=d8da5fa411639cfc ===
Gold sentences and facts:
Sentence 0:  His primary jockey is Tony McCoy.
  Fact 0.0: The unnamed subject of this sentence has Tony McCoy as its primary jockey.
  Fact 0.1: Tony McCoy is a primary jockey.
Sentence 1:  Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.
  Fact 1.0: Tony McCoy is based in Ireland and the UK.
  Fact 1.1: Tony McCoy rode a record 4,358 winners.
  Fact 1.2: Tony McCoy was Champion Jockey a record 20 consecutive times.
  Fact 1.3: Tony McCoy was Champion Jockey every year he was a professional.

Retrieved items:
<item 1>
Fred Winter
Frederick Thomas Winter CBE (20 September 1926 – 5 April 2004) was a British National Hunt racing racehorse jockey and trainer.

 He was British jump racing Champion Jockey four times and British jump racing Champion Trainer eight times.

 He is the only person to have won the Cheltenham Gold Cup, Champion Hurdle and Grand National as both jockey and trainer.

 Winter won the Grand National four times, as a jockey in 1957 (Sundew) and 1962 (Kilmore), and as a trainer in 1965 (Jay Trump) and 1966 (Anglo).
</item 1>

<item 2>
Irish jump racing Champion Jockey
The Champion Jockey of National Hunt racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1947.

 The current champion, as of 2017, is Ruby Walsh.
</item 2>

<item 3>
Binocular (horse)
Binocular, foaled on March 17, 2004 in France, is a French thoroughbred retired racehorse, born to sire, Enrique and dam Bleu Ciel Et Blanc.

 He is owned by J. P. McManus and trained by Nicky Henderson.

 His primary jockey is Tony McCoy.
</item 3>

<item 4>
German flat racing Champion Jockey
The Champion Jockey of flat racing in Germany is the jockey who has ridden the most winning horses during a season.

 The list of the Champion Jockey started 1870.

 The list below shows the Champion Jockey for each year since 1984.
</item 4>

<item 5>
British flat racing Champion Jockey
The Champion Jockey of flat racing in Great Britain is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey and the number of winners for each year since 1840.

 The seasonal record of jockeys' winners was published for the first time in 1846.

 The championship was sponsored for the first time in 2009 by online casino 32Red and is currently sponsored by Stobart Group.

 The 2016 title was determined by the number of winners ridden between 30 April and 15 October, in both turf and all-weather racing.

 Prior to 2015 the title was decided over the full length of the Flat turf season, from March to November, and carried no prize money to the winner.

 A prize of £25,000 to the champion jockey was also introduced as part of the 2015 changes.
</item 5>

<item 6>
Bart Cummings
James Bartholomew Cummings {'1': ", '2': ", '3': ", '4': "} (14 November 1927 – 30 August 2015), also known by his initials J. B. Cummings, was one of the most
successful Australian racehorse trainers.

 He was known as the "Cups King", referring to the Melbourne Cup, as he won 'the race that stops a nation' a record twelve times.

 During his lifetime Cummings was considered an Australian cultural icon and an Australian National Living Treasure.
</item 6>

<item 7>
Ernest Piggott was a three-time champion jockey and Grand National winner; his family included jockeys Keith and 11-time British flat champion Lester. Gordon
Richards won 26 British flat titles and was the only flat jockey knighted. Champion titles recognize most seasonal winners; Tony McCoy set records with 4,358
wins and 20 consecutive championships. Buster Parnell was Ireland’s 1969 flat champion. Fred Winter was a four-time jump champion and eight-time champion
</item 7>

<item 8>
Tony McCoy
Sir Anthony Peter McCoy, OBE (born 4 May 1974), commonly known as A. P. McCoy or Tony McCoy, is a Northern Irish former horse racing jockey.

 Based in Ireland and the UK, McCoy rode a record 4,358 winners, and was Champion Jockey a record 20 consecutive times, every year he was a professional.

 He stands 1.78 m (5'10"), far taller than most jockeys.
</item 8>

<item 9>
Gordon Richards
Sir Gordon Richards (5 May 1904 – 10 November 1986) was an English jockey.

 He was the British flat racing Champion Jockey 26 times and is often considered the world's greatest ever jockey.

 He remains the only flat jockey to have been knighted.
</item 9>

<item 10>
Irish flat racing Champion Jockey
The Champion Jockey of flat racing in Ireland is the jockey who has ridden the most winning horses during a season.

 The list below shows the Champion Jockey for each year since 1950.
</item 10>
=== END ITEM id=d8da5fa411639cfc ===

=== ITEM id=0df6848a20427c74 ===
Gold sentences and facts:
Sentence 0:  It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).
  Fact 0.0: The company began in 1995.
  Fact 0.1: The company began as the merger of two take-and-bake pizza companies.
  Fact 0.2: One of the merged companies was Papa Aldo's Pizza.
  Fact 0.3: Papa Aldo's Pizza was founded in 1981.
  Fact 0.4: The other merged company was Murphy's Pizza.
  Fact 0.5: Murphy's Pizza was founded in 1981.
Sentence 1: Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.
  Fact 1.0: Gino's Pizza and Spaghetti is a restaurant chain.
  Fact 1.1: Gino's Pizza and Spaghetti has 40 locations.
  Fact 1.2: Most of Gino's Pizza and Spaghetti's locations are within the U.S. state of West Virginia.
Sentence 2:  The company was founded by Kenney Grant in 1961.
  Fact 2.0: The company was founded by Kenney Grant.
  Fact 2.1: The company was founded in 1961.

Retrieved items:
<item 1>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 1>

<item 2>
Tudor's Biscuit World
Tudor's Biscuit World is a restaurant chain based in Huntington, West Virginia, most commonly found in West Virginia.

 Many West Virginia locations share a building with Gino's Pizza and Spaghetti, although the chain is more extensive than Gino's (which is exclusive to West
Virginia), having locations in southern Ohio, eastern Kentucky, and southwestern Virginia.

 In 2016 a franchise was opened in Panama City, Florida.

 Tudor's serves biscuits, biscuit sandwiches, homestyle breakfasts and dinners, muffins, and several side dishes.

 The chain was originally based in Charleston, West Virginia and many of the biscuit sandwiches are named for sports teams of interest in that area, including
teams at Marshall University, West Virginia University, and The University of Charleston.
</item 2>

<item 3>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 3>

<item 4>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 4>

<item 5>
Papa Murphy's Park
Papa Murphy's Park (formerly known as the Cal Expo Multi-Use Sports Field Facility and Bonney Field) is a sports venue located on the grounds of Cal Expo in
Sacramento, California.

 The soccer-specific stadium has a capacity of 11,569 and includes a full-sized (120 x 80 yard) soccer field.

 Papa Murphy's Park is the current home of Sacramento Republic FC soccer team and former home of PRO Rugby team, Sacramento Express.
</item 5>

<item 6>
Gino's Pizza and Spaghetti
Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.

 The company was founded by Kenney Grant in 1961.

 Many locations are shared with Tudor's Biscuit World although the Gino's brand is exclusive to West Virginia.

 There is one located in Ohio, while there are stand alone Tudor's locations in eastern Kentucky, southern Ohio and southwest Virginia.

 Gino's serves pizza, spaghetti, sandwiches, and more.

 Company headquarters are located in Huntington, West Virginia and Nitro, West Virginia.
</item 6>

<item 7>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 7>

<item 8>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 8>

<item 9>
Papa Gino's
Papa Gino's, Inc. is a restaurant chain based in Dedham, Massachusetts specializing in American-style pizza along with pasta, subs, salads, and a variety of
appetizers.

 There are over 150 Papa Gino's locations in Connecticut, Maine, Massachusetts, New Hampshire, and Rhode Island.
</item 9>

<item 10>
Gino's East
Gino's East is a Chicago-based restaurant chain, notable for its deep-dish pizza (sometimes called Chicago-style pizza), and for its interior walls, which
patrons have covered in graffiti and etchings.

 The restaurant features deep-dish pizza baked in cast-iron pans, as well as sandwiches, soups and salads.
</item 10>
=== END ITEM id=0df6848a20427c74 ===

=== ITEM id=b60a1ce8c0a286bf ===
Gold sentences and facts:
Sentence 0: Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.
  Fact 0.0: Adam Fox lived from 1883 to 1977.
  Fact 0.1: Adam Fox was a Canon.
  Fact 0.2: Adam Fox was the Dean of Divinity at Magdalen College, Oxford.
Sentence 1:  He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.
  Fact 1.0: Adam Fox was one of the first members of the Inklings.
  Fact 1.1: The Inklings was a literary group.
  Fact 1.2: The Inklings also included C.S. Lewis.
  Fact 1.3: The Inklings also included J.R.R. Tolkien.
Sentence 2: The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and late 1949.
  Fact 2.0: The Inklings was an informal literary discussion group.
  Fact 2.1: The Inklings was associated with the University of Oxford, England.
  Fact 2.2: The Inklings lasted for nearly two decades.
  Fact 2.3: The Inklings operated between the early 1930s and late 1949.

Retrieved items:
<item 1>
James F. Light
James F. Light was an American literary scholar, university vice president, and provost.

 During his academic career, he helped revive the works of satirist Nathanael West, with the first book length critical study of his work, "Nathanael West: An
Interpretive Study", (Northwestern Univ.

 Press, 1961).

 He was also the leading authority on John William De Forest, the early American realist whose work he critiqued in "John William De Forest" (Twayne Pub.,
1965), and he wrote extensively on J. D. Salinger, Robert Penn Warren and others.
</item 1>

<item 2>
Falmouth University
Falmouth University (Cornish: Pennskol Aberfala ) is a specialist University for the creative industries based in Falmouth and Penryn, Cornwall, England.

 Founded in 1902, it had previously been the Falmouth School of Art, Falmouth College of Art and Design and then Falmouth College of Arts until it received
taught degree-awarding powers, and the right to use the title "University College", in March 2005.

 In April 2008, University College Falmouth merged with Dartington College of Arts, adding a range of Performance courses to its portfolio.

 On 27 November 2012, a communication was released to the staff and students and local press that 'University College Falmouth is to be granted full university
status in a move that will further its ambition to become one of the top five arts universities in the world.'

 On 9 December 2012, the University College was officially granted full university status by the Privy Council.

 Falmouth University was judged by The Sunday Times to be the UK’s top arts university from 2015-2017.
</item 2>

<item 3>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 3>

<item 4>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 4>

<item 5>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 5>

<item 6>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 6>

<item 7>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 7>

<item 8>
Inklings
The Inklings was an informal literary discussion group associated with the University of Oxford, England, for nearly two decades between the early 1930s and
late 1949.

 The Inklings were literary enthusiasts who praised the value of narrative in fiction and encouraged the writing of fantasy.
</item 8>

<item 9>
Eagle Street College
The Eagle Street College was an informal literary society established in 1885 at the home of James William Wallace in Eagle Street, Bolton, to read and discuss
literary works, particularly the poetry of Walt Whitman, (1819–91).

 The group subsequently became known as the Bolton Whitman Fellowship or Whitmanites.

 Its founder members were Wallace, Dr John Johnston and Fred Wild.

 The group held an annual 'Whitman Day' celebration around the 31 May which was the poet's birthday.
</item 9>

<item 10>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 10>
=== END ITEM id=b60a1ce8c0a286bf ===

=== ITEM id=dc91da742a5e7f6f ===
Gold sentences and facts:
Sentence 0: Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.
  Fact 0.0: Return to Never Land is also known as Peter Pan 2.
  Fact 0.1: Return to Never Land is also known as Peter Pan In: Return to Never Land.
  Fact 0.2: Return to Never Land is a 2002 American film.
  Fact 0.3: Return to Never Land is an animated musical fantasy-adventure film.
  Fact 0.4: Return to Never Land was produced by Walt Disney Television Animation.
  Fact 0.5: Return to Never Land was released by Walt Disney Pictures.
  Fact 0.6: Return to Never Land was distributed by Walt Disney Studios Motion Pictures.
Sentence 1:  The film was co-produced by Walt Disney Pictures and Mayhem Pictures.
  Fact 1.0: Return to Never Land was co-produced by Walt Disney Pictures.
  Fact 1.1: Return to Never Land was co-produced by Mayhem Pictures.

Retrieved items:
<item 1>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 1>

<item 2>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 2>

<item 3>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 3>

<item 4>
Peter Pan: Adventures in Never Land
Peter Pan: Adventures in Never Land (also known as Peter Pan in Disney's Return to Never Land) is a 2002 video game based on the Peter Pan Disney franchise.
</item 4>

<item 5>
Never Land Books
The Never Land Books or Never Land Adventures are a series of short chapter books set in Never Land, the home of Peter Pan.

 They are based on the situations and characters established in the novel "Peter and the Starcatchers" and its sequels.

 Like the novels, they are written by Pulitzer Prize-winning humorist Dave Barry and suspense novelist Ridley Pearson, and illustrated by Greg Call.

 Although five books were planned, only three were published, in 2006–2008.

 The stories focus on supporting characters from the novels, such as the Indians, mermaids, pirates, and Lost Boys.

 They include:
</item 5>

<item 6>
Home on the Range (2004 film)
Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.

 The 45th Disney animated feature film, it was the last 2D animated Disney film released until "The Princess and the Frog" in 2009.

 Named after the popular country song of the same name, "Home on the Range" features the voices of Roseanne Barr, Judi Dench, Jennifer Tilly, Cuba Gooding Jr.,
Randy Quaid, and Steve Buscemi.

 The film is set in the Old West, and centers on a mismatched trio of dairy cows—brash, adventurous Maggie; prim, proper Mrs. Caloway; and ditzy, happy-go-lucky
Grace.

 The three cows must capture an infamous cattle rustler named Alameda Slim for his bounty in order to save their idyllic farm from foreclosure.

 Aiding them in their quest is Lucky Jack, a feisty, peg-legged rabbit, but a selfish horse named Buck, eagerly working in the service of Rico, a famous bounty
hunter, seeks the glory for himself.
</item 6>

<item 7>
Us (film)
Us is a 1991 television movie broadcast on CBS, produced, written and directed by Michael Landon.

 Landon also starred in the film, along with Barney Martin and Casey Peterson.

 It was a pilot for what would have been Landon's fourth consecutive television series; Landon's death that year precluded its going ahead.

 Landon played Jeff Hayes, a man just released from prison after serving many years due to being wrongfully convicted of killing a wealthy man's wife.
</item 7>

<item 8>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 8>

<item 9>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 9>

<item 10>
Summer Magic
Summer Magic is a 1963 Walt Disney Productions film starring Hayley Mills, Burl Ives, and Dorothy McGuire in a story about a Boston widow and her children
taking up residence in a small town in Maine.

 The film was based on the novel "Mother Carey's Chickens" by Kate Douglas Wiggin and was directed by James Neilson.

 This was the fourth of six film Mills did for Disney, and the young actress received a Golden Globe nomination for her work here.
</item 10>
=== END ITEM id=dc91da742a5e7f6f ===

=== ITEM id=65c01122cc22193b ===
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
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 1>

<item 2>
Vaněk plays
The Vaněk plays are a set of plays to which the character Ferdinand Vaněk is central.

 Vaněk first appeared in the play "Audience" by Václav Havel.

 He subsequently appeared in three other plays by Havel ("Protest", "Unveiling", and "Dozens of Cousins"), as well as plays by his friends and colleagues,
including Pavel Landovský and Tom Stoppard.
</item 2>

<item 3>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 3>

<item 4>
Václav Havel Tribute Concert
Tribute to Václav Havel (Czech: Pocta Václavu Havlovi) was an event held in memory of Václav Havel, the last Czechoslovak and the first Czech President, writer,
playwright and human rights activist.

 The concert took place in Lucerna Music Bar in Prague on 23 December 2011.

 Václav Havel died on Sunday 18 December 2011 at 10:15 a.m.
</item 4>

<item 5>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 5>

<item 6>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 6>

<item 7>
Letters to Olga
Letters to Olga ("Czech:Dopisy Olze") is a book of compiled letters written by Czech playwright, dissident, and future president, Václav Havel to his wife Olga
Havlová during his nearly four-year imprisonment from May 1979 to March 1983.

 (Havel was released when he came down with a high fever and received a medical discharge).

 Havel was imprisoned by the communist regime of then Czechoslovakia for being one of the leaders of The Committee for the Defense of the Unjustly Prosecuted
(VONS) - most of whom had been signatories of the human rights document Charter 77.
</item 7>

<item 8>
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
</item 8>

<item 9>
Václav Havel Human Rights Prize
The Václav Havel Human Rights Prize is an annual €60,000 award established in 2013 by the Parliamentary Assembly of the Council of Europe, the Václav Havel
Library and the Charta 77 Foundation, to honour "outstanding" civil society action in defence of human rights, in Europe and beyond.
</item 9>

<item 10>
Truth prevails
"Truth prevails" (Czech: "Pravda vítězí" , Slovak: "Pravda víťazí" , Latin: "Veritas vincit" ) was a motto inscribed on the banner of the President of
Czechoslovakia, still used today on the banner of the President of the Czech Republic.

 The banner of the President is one of the national symbols according to the Czech Constitution.

 The phrase also appears along the base of the Jan Hus Memorial in Prague.

 The motto is believed to be derived from Jan Hus' phrase "Seek the truth, hear the truth, learn the truth, love the truth, speak the truth, hold the truth and
defend the truth until death".

 "Truth prevails" was adopted as motto by the first President of Czechoslovakia Tomáš Garrigue Masaryk in 1918 and then echoed in Václav Havel's notion of "life
in truth" and in his famous statement "Truth and love must prevail over lies and hatred" (Czech: "Pravda a láska musí zvítězit nad lží a nenávistí" ).
</item 10>
=== END ITEM id=65c01122cc22193b ===

=== ITEM id=caa649a109bb35b5 ===
Gold sentences and facts:
Sentence 0: Robert Harry Philibosian (born 1940) is an American politician.
  Fact 0.0: Robert Harry Philibosian was born in 1940.
  Fact 0.1: Robert Harry Philibosian is an American politician.
Sentence 1: David R. Ignatius (May 26, 1950), is an American journalist and novelist.
  Fact 1.0: David R. Ignatius was born on May 26, 1950.
  Fact 1.1: David R. Ignatius is an American journalist.
  Fact 1.2: David R. Ignatius is an American novelist.

Retrieved items:
<item 1>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 1>

<item 2>
David I. Walsh
David Ignatius Walsh (November 11, 1872June 11, 1947) was a United States politician from Massachusetts.

 A member of the Democratic Party, he served as the 46th Governor of Massachusetts before serving several terms in the United States Senate.
</item 2>

<item 3>
Robert Philibosian
Robert Harry Philibosian (born 1940) is an American politician.

 He served as Los Angeles County District Attorney from 1982 to 1984.

 He received his B.A. in history from Stanford University and his law degree from Southwestern Law School, and was admitted to the California State Bar in 1968.

 He is now Of counsel at the law firm of Sheppard, Mullin, Richter & Hampton.
</item 3>

<item 4>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 4>

<item 5>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 5>

<item 6>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 6>

<item 7>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 7>

<item 8>
David Smick
David M. Smick is a macroeconomic adviser to a select group of prominent global investment funds and a non-fiction author.

 He is the chairman and CEO of Johnson Smick International, a financial advisory firm in Washington, D.C. where he is in partnership with former Federal Reserve
Vice Chairman Manuel H. Johnson.

 Smick wrote an acclaimed book "The World Is Curved" on the financial perils of globalization, in response to "The World Is Flat" bestseller by Thomas Friedman.

 The book received positive reviews by the "USA Today" and "Forbes", was called "astonishingly prescient" by David Brooks and "clairvoyant" by David Ignatius.
</item 8>

<item 9>
Richard Ziser
Richard Ziser is a Nevada Real Estate Investor, Socially Conservative Political activist and U.S. Republican Politician.

 He was born June 7, 1953, in Pomona, CA., and has resided in Las Vegas Nevada since 1991.

 He graduated from California State Polytechnic University, Pomona (Cal Poly Pomona) with a BS in Industrial Engineering, 1976; then subsequently from Simon
Greenleaf University in Santa Ana, Ca.

 (now a campus of Trinity International University, with an MA in Christian Apologetics in 1989.
</item 9>

<item 10>
Mark Krikorian
Mark Krikorian has been the executive director of the Center for Immigration Studies, a think-tank in Washington, D. C., since 1995.

 The Center describes itself as an "independent, non-partisan research organization" in Washington, D. C., that examines and critiques the impact of immigration
on the United States.

 Animated by a "pro-immigrant, low-immigration vision which seeks fewer immigrants, but a warmer welcome for those admitted", the Center was established in 1985
to provide immigration research.

 Krikorian is a regular contributor to the conservative publication "National Review", and is a regular participant at "National Review Online's" "The Corner."
</item 10>
=== END ITEM id=caa649a109bb35b5 ===

=== ITEM id=da764e87d5e3e23d ===
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
Maria Theresa (disambiguation)
Maria Theresa or Maria Theresia most often refers to Empress Maria Theresa (1717-1780), Queen of Hungary and Bohemia, Archduchess of Austria and ruler of the
rest of the Habsburg Empire (1740-1780).
</item 1>

<item 2>
Gottlieb Muffat
Gottlieb Muffat (April 1690 – 9 December 1770), son of Georg Muffat, served as "Hofscholar" under Johann Fux in Vienna from 1711 and was appointed to the
position of third court organist at the "Hofkapelle" in 1717.

 He acquired additional duties over time including the instruction of members of the Imperial family, among them the future Empress Maria Theresa.

 He was promoted to second organist in 1729 and first organist upon the accession of Maria Theresa to the throne in 1741.

 He retired from official duties at the court in 1763.
</item 2>

<item 3>
Operetta (film)
Operetta (German: Operette) is a 1940 musical film directed by Willi Forst and starring Forst, Maria Holst and Dora Komar.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It is the first film in director Willi Forst's "Viennese Trilogy" followed by "Vienna Blood" (1942) and "Viennese Girls" (1945).

 The film portrays the life of Franz Jauner (1832–1900), a leading musical figure in the city.

 It is both an operetta film and a Wiener Film.
</item 3>

<item 4>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 4>

<item 5>
Maria Theresa of Austria (1801–1855)
Maria Theresa of Austria (21 March 1801 – 12 January 1855) was born an Archduchess of Austria and Princess of Tuscany.

 She was a daughter of Ferdinand III, Grand Duke of Tuscany, and Luisa of Naples and Sicily.

 She was named after her double great grandmother Empress Maria Theresa.

 In 1817, she married Charles Albert of Sardinia and subsequently became the Queen of Sardinia upon her husband's accession to the throne in 1831.
</item 5>

<item 6>
Paride ed Elena
Paride ed Elena (] ; "Paris and Helen") is an opera by Christoph Willibald Gluck.

 It is the third of Gluck's so-called reform operas for Vienna, following "Alceste" and "Orfeo ed Euridice", and the least often performed of the three.

 Like its predecessors, the libretto was written by Ranieri de' Calzabigi.

 The opera tells the story of the events between the Judgment of Paris and the flight of Paris and Helen to Troy.

 It was premiered at the Burgtheater in Vienna on 3 November 1770.
</item 6>

<item 7>
Maria Theresa thaler
The Maria Theresa thaler (MTT) is a silver bullion coin that has been used in world trade continuously since they were first minted in 1741, at that time using
the then Reichsthaler standard of 9 thalers to the Vienna mark.

 In 1750 the thaler was debased to 10 thalers to the Vienna Mark (a weight approximating half a pound of fine silver).

 The following year the new standard was effectively adopted across the German-speaking world when that standard was accepted formally in the Bavarian monetary
convention.

 It is owing to the date of the Bavarian Monetary convention that many writers erroneously state that the Maria Theresa Thaler was first struck in 1751.

 It was named after Empress Maria Theresa, who ruled Austria, Hungary, and Bohemia from 1740 to 1780.

 The word "thaler" gave rise to "daalder" and "daler", which became "dollar" in English.
</item 7>

<item 8>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 8>

<item 9>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 9>

<item 10>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 10>
=== END ITEM id=da764e87d5e3e23d ===

=== ITEM id=ac8545ad84312be6 ===
Gold sentences and facts:
Sentence 0:  It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
  Fact 0.0: The book is a companion book to 'A Wrinkle in Time'.
  Fact 0.1: The book is part of the Time Quintet.
Sentence 1: The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
  Fact 1.0: The Time Quintet is a fantasy/science fiction series.
  Fact 1.1: The Time Quintet is a series of five young adult novels.
  Fact 1.2: The Time Quintet was written by Madeleine L'Engle.

Retrieved items:
<item 1>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 1>

<item 2>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 2>

<item 3>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 3>

<item 4>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 4>

<item 5>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 5>

<item 6>
The Four Winds of Love
The Four Winds of Love is the overall title for a series of six novels written by Compton Mackenzie, "The East Wind of Love" (1937), "The South Wind of Love"
(1938), "The West Wind of Love" (1940), "West to North" (1942), "The North Wind of Love, Book 1" (1944) and "The North Wind of Love, Book 2" (1945), which taken
together constitute a major fictional chronicle of the first forty years of the twentieth century.

 The main protagonist of the hexalogy is the semi-autobiographical character of John Ogilvie.
</item 6>

<item 7>
Time Quintet
The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
</item 7>

<item 8>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 8>

<item 9>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 9>

<item 10>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 10>
=== END ITEM id=ac8545ad84312be6 ===

=== ITEM id=c77d2d7afd896408 ===
Gold sentences and facts:
Sentence 0: The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.
  Fact 0.0: The Shins are an American indie rock band.
  Fact 0.1: The Shins are from Albuquerque, New Mexico.
  Fact 0.2: The Shins were formed in 1996.
Sentence 1: Paramore is an American rock band from Franklin, Tennessee, formed in 2004.
  Fact 1.0: Paramore is an American rock band.
  Fact 1.1: Paramore is from Franklin, Tennessee.
  Fact 1.2: Paramore was formed in 2004.

Retrieved items:
<item 1>
Paramore discography
The American rock band Paramore has released five studio albums, three extended plays, two live albums, eighteen singles, one video album, and nineteen music
videos.

 The band was formed in Franklin, Tennessee, in 2004 by lead vocalist Hayley Williams with guitarists Josh Farro and Taylor York, bassist Jeremy Davis, and
drummer Zac Farro.

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling".

 Three singles were released from the album, but none of them charted.

 The album did not chart in the "Billboard" 200 either, although it peaked at number thirty in the "Billboard" Top Heatseekers.

 "All We Know Is Falling" received Gold certification in the United Kingdom and in July 2014 the RIAA certified the album Gold in the United States.
</item 1>

<item 2>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 2>

<item 3>
List of songs recorded by Paramore
American rock band Paramore have recorded songs for five studio albums, a box set, an extended play and two soundtrack albums.

 In 2002, at age 13, vocalist Hayley Williams moved to Franklin, Tennessee, where she met brothers Josh Farro and Zac Farro.

 The band was officially formed by Josh Farro (lead guitar and backing vocals), Zac Farro (drums), Jeremy Davis (bass guitar) and Williams (lead vocals) in
2004, with the later addition of Williams' neighbor Jason Bynum (rhythm guitar).

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling" that year.

 Three singles were released to promote the album.
</item 3>

<item 4>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 4>

<item 5>
Hate to See Your Heart Break
"Hate to See Your Heart Break" is a song by American rock band Paramore, recorded for their 2013 self-titled fourth album "Paramore".

 It was re-recorded to feature vocals by Joy Williams (formerly of The Civil Wars) for the 2014 deluxe edition of the album, the first time Paramore has
collaborated with another artist or group on a studio recording.
</item 5>

<item 6>
Singles Club (EP)
Singles Club is an EP released by American rock band Paramore.

 The songs were released as promotional singles between October and December 2011, culminating in the release of a box set containing the three constituent
songs plus "Monster", which was recorded during the same sessions.

 The EP and box set were released on December 14, 2011 on Paramore's website.

 The EP is the first release of new material by Paramore not to include band members Josh and Zac Farro.
</item 6>

<item 7>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 7>

<item 8>
The Only Exception
"The Only Exception" is a song by American rock band Paramore.

 It was released by Fueled by Ramen in February 2010 as the third single from the band's third studio album, "Brand New Eyes" (2009).

 The song was written by band members Hayley Williams and Josh Farro; Paramore is also credited as being co-producers to the song.

 The song was generally well received by music critics; praise of the song was mainly about Williams' vocal performance.

 Music critics reviewing the song noted that "The Only Exception" was a different musical theme for the band.
</item 8>

<item 9>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 9>

<item 10>
Paramore (album)
Paramore is the self-titled fourth studio album by American rock band Paramore.

 It was released on April 5, 2013, through Fueled by Ramen as a follow-up to "Brand New Eyes" (2009).

 It is their first full-length album released after the departure of co-founders Josh and Zac Farro in 2010.

 Recorded between April and November 1, 2012, the album was described by the band as being a "statement" and a reintroduction of the band to the world and to
themselves.

 It is the only album without Zac Farro on drums, and the final album recorded with bassist Jeremy Davis before he left the band in 2015.
</item 10>
=== END ITEM id=c77d2d7afd896408 ===

=== ITEM id=a97736eb1ae7f57f ===
Gold sentences and facts:
Sentence 0:  Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
  Fact 0.0: The unnamed person in this passage has other voice-over roles
  Fact 0.1: The unnamed person voiced Dr. Drakken in Kim Possible
  Fact 0.2: The unnamed person voiced Motor Ed in Kim Possible
  Fact 0.3: The unnamed person voiced Brother Blood in Teen Titans
  Fact 0.4: The unnamed person voiced Aquaman in an unnamed work
  Fact 0.5: The unnamed person voiced Rico in The Penguins of Madagascar
  Fact 0.6: The unnamed person voiced Ogre in an unnamed work
  Fact 0.7: The unnamed person voiced Niblet in Pound Puppies
  Fact 0.8: The unnamed person voiced the Scotsman in Samurai Jack
  Fact 0.9: The unnamed person voiced Shnitzel in Chowder
Sentence 1: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 1.0: Chowder is an American animated television series
  Fact 1.1: Chowder was created by C. H. Greenblatt
  Fact 1.2: Chowder was created for Cartoon Network

Retrieved items:
<item 1>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 1>

<item 2>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 2>

<item 3>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 3>

<item 4>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 4>

<item 5>
Harvey Beaks
Harvey Beaks is an American animated television series created by C.H. Greenblatt for Nickelodeon.

 The series premiered on March 28, 2015 following the 2015 Kids' Choice Awards.
</item 5>

<item 6>
Adventure Time (pilot)
"Adventure Time" is an animated short created by Pendleton Ward, as well as the pilot to the Cartoon Network series of the same name.

 The short follows the adventures of Pen (voiced by Zack Shada), a human boy, and his best friend Jake (voiced by John DiMaggio), a dog with magical powers to
change shape and grow and shrink at will.

 In this episode, Pen and Jake have to rescue Princess Bubblegum (voiced by Paige Moss) from the antagonistic Ice King (voiced by John Kassir).
</item 6>

<item 7>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 7>

<item 8>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 8>

<item 9>
List of Invader Zim characters
"Invader Zim" is an American animated television series created by Jhonen Vasquez and originally aired on Nickelodeon.

 The recurring cast includes voice actors Richard Steven Horvitz, Rodger Bumpass, Kevin McDonald ("The Kids in the Hall") and John de Lancie ("").

 Vasquez voices parts in the show.

 The show was cancelled early in its run and some episodes were unfinished.

 The show ran for two seasons before its cancellation.
</item 9>

<item 10>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 10>
=== END ITEM id=a97736eb1ae7f57f ===

=== ITEM id=de291b59d198d4b0 ===
Gold sentences and facts:
Sentence 0:  He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.
  Fact 0.0: The person described in this sentence was the brother of Alfred Jodl.
  Fact 0.1: Alfred Jodl was Chief of the Operations Staff of the OKW.
Sentence 1: Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
  Fact 1.0: Alfred Josef Ferdinand Jodl lived from 10 May 1890 to 16 October 1946.
  Fact 1.1: Alfred Jodl was a German general.
  Fact 1.2: Alfred Jodl was a war criminal during World War II.
  Fact 1.3: Alfred Jodl served as Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).

Retrieved items:
<item 1>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 1>

<item 2>
Hasso von Wedel (general)
Hasso von Wedel (20 November 1898 – 3 January 1961) was a German general who commanded the Wehrmacht Propaganda Troops during World War II.

 He was directly subordinate to the head of OKW Operations Staff ("Wehrmachtführungsstab, WFSt.")

, General Alfred Jodl.

 Wedel's Propaganda Department had control over the propaganda units and served to mediate between them and the Reich Propaganda Ministry of Joseph Goebbels.
</item 2>

<item 3>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 3>

<item 4>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 4>

<item 5>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 5>

<item 6>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 6>

<item 7>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 7>

<item 8>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 8>

<item 9>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 9>

<item 10>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 10>
=== END ITEM id=de291b59d198d4b0 ===

=== ITEM id=16e69059aa1991e7 ===
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
Warren Hound
Warren Hound refers to the group of Mediterranean hare-hunting dogs.

 In Spanish, this kind of breeds are called "podencos" and in Portuguese, "podengos".

 Generally speaking, a warren hound has erect ears, a smooth (or in some cases, wire) coat and a whip-like tail.

 However, there are some exceptions: for instance, the Cretan Hound has usually rose-ears and a curly or curved tail, while the Barrocal Algarvio is
long-haired.

 Unlike hounds or sighthounds which rely on just one sense, warren hounds tend to work with three senses: smell, sight and hearing.
</item 4>

<item 5>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
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
=== END ITEM id=16e69059aa1991e7 ===
