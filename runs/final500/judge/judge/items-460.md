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

=== ITEM id=8f05de56eb803653 ===
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
Kenny Ortega
Kenneth John "Kenny" Ortega (born April 18, 1950) is an American producer, director, and choreographer.

 He is best known for directing "Hocus Pocus", the "High School Musical" trilogy, "Descendants" and Michael Jackson's "This Is It" concerts.
</item 1>

<item 2>
Mark Lee Ping-bing
Mark Lee Ping-bing (李屏賓; born August 8, 1954) is a Taiwanese cinematographer, photographer and author with over 70 films and 21 international awards to his
credit including 2 Glory Of The Country Awards from the Government Information Office of Taiwan and the president of Taiwan's Light Of The Cinema Award.

 Lee began his film career in 1977 and in 1985 he started his prolific collaboration with Taiwanese filmmaker Hou Hsiao-hsien.

 Known best for his use of natural lighting utilizing real film and graceful camera movement, Lee received the Grand Technical Prize at the Cannes Film Festival
in 2000 for "In the Mood for Love".

 A member of the Academy of Motion Picture Arts and Sciences, Lee was honored with nominations by the American Society of Cinematographers for its 2014 First
Annual Spotlight Award for Best Cinematography for his work on the 2012 film "Renoir" and by the French Academy of Cinema Arts for a Cesar Award for Best
Cinematography in 2014 also for the film "Renoir".
</item 2>

<item 3>
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
</item 3>

<item 4>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 4>

<item 5>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 5>

<item 6>
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
</item 6>

<item 7>
The Assassin (2015 film)
The Assassin (; or: "The Assassin Niè Yǐnniáng") is a 2015 wuxia film directed by Taiwanese director Hou Hsiao-hsien.

 A Taiwan/China/Hong Kong co-production, it was an official selection in the main competition section at the 2015 Cannes Film Festival.

 At Cannes, Hou won the award for Best Director.

 It was released in China and Hong Kong on 27 August, and a day later in Taiwan on 28 August 2015.

 It was selected as the Taiwanese entry for the Best Foreign Language Film at the 88th Academy Awards but it was not nominated.
</item 7>

<item 8>
Academy Honorary Award
The Academy Honorary Award is often awarded in preference to those with noted achievements in motion pictures who have nevertheless never won an Academy Award.

 Thus, many of its recipients are Classic Hollywood stars, such as Lillian Gish, Barbara Stanwyck, Kirk Douglas, and Lauren Bacall.
</item 8>

<item 9>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 9>

<item 10>
Hou Hsiao-hsien
Hou Hsiao-hsien (] born 8 April 1947) is a Taiwanese film director, screenwriter, producer and actor.

 He is a leading figure in world cinema and in Taiwan's New Wave cinema movement, an auteur.

 He won the Golden Lion at the Venice Film Festival in 1989 for his film "A City of Sadness" (1989), and the Best Director award at the Cannes Film Festival in
2015 for "The Assassin" (2015).

 Other highly regarded works of his include "The Puppetmaster" (1993) and "Flowers of Shanghai" (1998).
</item 10>
=== END ITEM id=8f05de56eb803653 ===

=== ITEM id=5318cde0b66b5a52 ===
Gold sentences and facts:
Sentence 0: Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
  Fact 0.0: Alan Crosland was born on August 10, 1894.
  Fact 0.1: Alan Crosland died on July 16, 1936.
  Fact 0.2: Alan Crosland was an American stage actor.
  Fact 0.3: Alan Crosland was an American film director.
Sentence 1: Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.
  Fact 1.0: Kurt Voss was born Kurt Christopher Peter Wössner.
  Fact 1.1: Kurt Voss is an American film director.
  Fact 1.2: Kurt Voss is an American screenwriter.
  Fact 1.3: Kurt Voss is an American musician-songwriter.

Retrieved items:
<item 1>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 1>

<item 2>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 2>

<item 3>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 3>

<item 4>
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
</item 4>

<item 5>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 5>

<item 6>
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
</item 6>

<item 7>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 7>

<item 8>
Arthur Tedder, 1st Baron Tedder
After the war he served as Chief of the Air Staff, in which role he advocated increased recruiting in the face of many airmen leaving the service, doubled the
size of RAF Fighter Command and implemented arrangements for the Berlin Airlift in 1948.

 After the war he held senior positions in business and academia.
</item 8>

<item 9>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 9>

<item 10>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 10>
=== END ITEM id=5318cde0b66b5a52 ===

=== ITEM id=56306e237464b884 ===
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
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 1>

<item 2>
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
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 4>

<item 5>
Graph query row:
person: Waylon Jennings
entity: The Highwaymen
relationship: MEMBER_OF
role: None
period: None
status: None
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
There Ain't No Good Chain Gang
"There Ain't No Good Chain Gang" is a song written by Hal Bynum and Dave Kirby, and recorded by American country music artists Johnny Cash and Waylon Jennings.

 It was released in May 1978 as the second single from the album "I Would Like to See You Again".

 The song reached #2 on the "Billboard" Hot Country Singles & Tracks chart.
</item 8>

<item 9>
Graph query row:
person: Johnny Cash
entity: The Highwaymen
relationship: MEMBER_OF
role: None
period: None
status: None
</item 9>

<item 10>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 10>
=== END ITEM id=56306e237464b884 ===

=== ITEM id=2328b01345397529 ===
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
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 1>

<item 2>
Letters to Olga
Letters to Olga ("Czech:Dopisy Olze") is a book of compiled letters written by Czech playwright, dissident, and future president, Václav Havel to his wife Olga
Havlová during his nearly four-year imprisonment from May 1979 to March 1983.

 (Havel was released when he came down with a high fever and received a medical discharge).

 Havel was imprisoned by the communist regime of then Czechoslovakia for being one of the leaders of The Committee for the Defense of the Unjustly Prosecuted
(VONS) - most of whom had been signatories of the human rights document Charter 77.
</item 2>

<item 3>
Václav Havel Tribute Concert
Tribute to Václav Havel (Czech: Pocta Václavu Havlovi) was an event held in memory of Václav Havel, the last Czechoslovak and the first Czech President, writer,
playwright and human rights activist.

 The concert took place in Lucerna Music Bar in Prague on 23 December 2011.

 Václav Havel died on Sunday 18 December 2011 at 10:15 a.m.
</item 3>

<item 4>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 4>

<item 5>
Truth prevails
"Truth prevails" (Czech: "Pravda vítězí" , Slovak: "Pravda víťazí" , Latin: "Veritas vincit" ) was a motto inscribed on the banner of the President of
Czechoslovakia, still used today on the banner of the President of the Czech Republic.

 The banner of the President is one of the national symbols according to the Czech Constitution.

 The phrase also appears along the base of the Jan Hus Memorial in Prague.

 The motto is believed to be derived from Jan Hus' phrase "Seek the truth, hear the truth, learn the truth, love the truth, speak the truth, hold the truth and
defend the truth until death".

 "Truth prevails" was adopted as motto by the first President of Czechoslovakia Tomáš Garrigue Masaryk in 1918 and then echoed in Václav Havel's notion of "life
in truth" and in his famous statement "Truth and love must prevail over lies and hatred" (Czech: "Pravda a láska musí zvítězit nad lží a nenávistí" ).
</item 5>

<item 6>
Czech presidential election, 1993
The 1993 Czech presidential election was held on 26 January 1993 to elect the first President of independent Czech Republic.

 The first election was done indirectly by the members of the Czech Republic Parliament.

 Václav Havel was elected President.

 The election was complicated only by attacks of Republicans against Havel and by bomb threat to the Parliament.
</item 6>

<item 7>
The Pig, or Václav Havel's Hunt for a Pig
The Pig, or Václav Havel's Hunt for a Pig is the final work by Václav Havel, co-authored by Vladimír Morávek.

 The English translation is by Edward Einhorn .

 Originally a short dialogue from 1987 (entitled simply "The Pig") and printed in a samizdat, the piece is a comic (and true) story of Václav Havel’s efforts to
hold a pig roast for his friends.
</item 7>

<item 8>
Václav Havel
Václav Havel (] ; 5 October 193618 December 2011) was a Czech writer and former dissident, who served as the 1st President of the Czech Republic from 1993 to
2003.

 Prior to that he also served as the last president of Czechoslovakia from 1989 until its dissolution in 1992.

 As a writer of Czech literature, he is known for his plays, essays, and memoirs.
</item 8>

<item 9>
Václav Havel Human Rights Prize
The Václav Havel Human Rights Prize is an annual €60,000 award established in 2013 by the Parliamentary Assembly of the Council of Europe, the Václav Havel
Library and the Charta 77 Foundation, to honour "outstanding" civil society action in defence of human rights, in Europe and beyond.
</item 9>

<item 10>
Aphra Behn
Aphra Behn ( ; 14 December 1640?

 (baptismal date)–16 April 1689) was a British playwright, poet, translator and fiction writer from the Restoration era.

 As one of the first English women to earn her living by her writing, she broke cultural barriers and served as a literary role model for later generations of
women authors.

 Rising from obscurity, she came to the notice of Charles II, who employed her as a spy in Antwerp.

 Upon her return to London and a probable
</item 10>
=== END ITEM id=2328b01345397529 ===

=== ITEM id=f1e96c2cbe624cfa ===
Gold sentences and facts:
Sentence 0:  The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI Studios in London, which was soon renamed Abbey Road Studios.
  Fact 0.0: The title and cover are an homage to the Beatles album Abbey Road.
  Fact 0.1: 926 East McLemore Avenue is the address of the Stax Studios in Memphis.
  Fact 0.2: Abbey Road was the address of EMI Studios in London.
  Fact 0.3: EMI Studios in London was soon renamed Abbey Road Studios.
Sentence 1:  This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
  Fact 1.0: This road is best known for the Abbey Road Studios.
  Fact 1.1: This road is best known for the 1969 album Abbey Road by The Beatles.

Retrieved items:
<item 1>
Temple Bar, London
The term "Temple Bar" strictly refers to a notional bar or barrier across the route, but is commonly used to refer to the 17th-century ornamental Baroque arched
gateway designed by Christopher Wren which spanned the road until its removal in 1878.

 Wren's arch was preserved and was re-erected in 2004 in the City, in Paternoster Square next to St Paul's Cathedral.
</item 1>

<item 2>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 2>

<item 3>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 3>

<item 4>
Temple Bar, London
Temple Bar was the principal ceremonial entrance to the City of London on its western side from the City of Westminster.

 It is situated on the historic royal ceremonial route from the Tower of London to the Palace of Westminster, the two chief residences of the mediaeval English
monarchs, and from the Palace of Westminster to St Paul's Cathedral.

 The road east of Temple Bar and within the City is Fleet Street, the road to the west, in Westminster, is The Strand.

 At Temple Bar the Corporation of the City of London formerly erected a barrier to regulate trade into the City.

 The 19th century Royal Courts of Justice are located next to it on its north side, having been moved from Westminster Hall.

 To its south is the Temple Church and the Inner Temple and Middle Temple Inns of Court.

 As the most important entrance to the City of London from Westminster, it was formerly long the custom for the monarch to halt at Temple Bar before entering
the City of London, in order for the Lord Mayor to offer the Corporation's pearl-encrusted Sword of State as a token of loyalty.
</item 4>

<item 5>
Nashville Terminal Subdivision
There are two crossings near each other, Williams Ave and Nesbitt Lane at Amqui.

 From here the Terminal goes south about 2 miles to the Nashville National Cemetery to the first major location, known as Ekin, 000179, where there is a cross
over track from number 1 to number 2 track (left to right track).

 There is also the first EDD (Defect Detector) at 000179.1.

 Few more miles southward, the next major location appears.

 At 000181.0, Maplewood is a major location for the Terminal.

 From here the regular Terminal goes south to swap crews at Kayne Ave, and the right side, Radnor Cutoff, detours the city and gives yard departures and
arrivals direct access to and from Radnor yard to cities like Louisville and Chicago.

 The Cutoff runs from Maplewood to Shelby Park double track.

 From Shebly the track converges into one to pass the historic Shelby bridge, then it opens back to double track.

 From there, the cutoff hits the Intermediates at 0BA187.0 known as Chicken Pike.
</item 5>

<item 6>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 6>

<item 7>
Nashville Terminal Subdivision
From here, the main continues south until the CR Cumberland River Swingbridge, where the main converges into a single track shortly to cross the bridge.

 At this point, trains had been running at track speed of around 40.

 From the drawbridge into town, speed is reduced to around 10.

 After the bridge is passed, the main returns to double track in downtown.

 On the #1 track about a half a mile south, another connecting track is present.

 This is the Wye track that connects the main with the Bruceton side, while rarely used for mainline trains, locals and river jobs use it.

 The location is known as 8th Avenue or 8th Avenue Wye.

 The main then runs down to Kayne Ave, the central hotbed of all Nashville thru traffic.

 The Memphis, Tennessee Ex-L&N Bruceton Subdivision meets with the Terminal.

 The Bruceton Sub begins at Church Street at 00N0.0.

 The line then runs single track until 00N0.7 "11th Avenue" where it turns into double track and also meets the aforementioned, Wye track.
</item 7>

<item 8>
Stax Museum of American Soul Music
The Stax Museum of American Soul Music is a museum located in Memphis, Tennessee, at 926 East McLemore Avenue, the former location of Stax Records.

 It is operated by Soulsville USA, which also operates the adjacent Stax Music Academy.
</item 8>

<item 9>
Basie on the Beatles
Basie on the Beatles is an album by pianist and bandleader Count Basie featuring performances recorded in late 1969 and released on the short-lived Happy Tiger
label.

 It was Basie's second album of Beatles' compositions following 1966's "Basie's Beatle Bag" and featured liner notes by Ringo Starr.
</item 9>

<item 10>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 10>
=== END ITEM id=f1e96c2cbe624cfa ===

=== ITEM id=b8e537162c5ce2f9 ===
Gold sentences and facts:
Sentence 0: The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of its kind in Northern Europe.
  Fact 0.0: The Cathedral and Metropolitical Church of Saint Peter in York is commonly known as York Minster.
  Fact 0.1: York Minster is the cathedral of York, England.
  Fact 0.2: York Minster is one of the largest cathedrals of its kind in Northern Europe.
Sentence 1: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.
  Fact 1.0: The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

Retrieved items:
<item 1>
Nashville Terminal Subdivision
This is where the Radnor A yard meets the main.

 There is a single departure/arrival track that meets the #2 track along with a crossover from #1 to #2 track.

 Commonly, the #2 track south of the signals is used to halt trains.

 This location is known as TVA, because of the power station that is adjacent.

 From here the Terminal runs about 2.5 miles south until we hit the southern tip.

 The tracks converge onto one single main, at 000/0BA196.6 known as Brentwood.

 Speed is increased to 50 and jurisdiction to the S.E. dispatcher.

 The right track is the main, S&NA North, while the left track is the Nashville Subdivision which runs to Columbia and exchanges freight with the TSRR.

 The Nashville Terminal Subdivision is one of the busiest locations on the CSX network, and one of the most important.
</item 1>

<item 2>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
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
York Minster
The Cathedral and Metropolitical Church of Saint Peter in York, commonly known as York Minster, is the cathedral of York, England, and is one of the largest of
its kind in Northern Europe.

 The minster is the seat of the Archbishop of York, the second-highest office of the Church of England, and is the mother church for the Diocese of York and the
Province of York.

 It is run by a dean and chapter, under the Dean of York.

 The title "minster" is attributed to churches established in the Anglo-Saxon period as missionary teaching churches, and serves now as an honorific title.

 Services in the minster are sometimes regarded as on the High Church or Anglo-Catholic end of the Anglican continuum.
</item 4>

<item 5>
Old Palace (York)
The Old Palace in the city of York, North Yorkshire, England, is also known as the Minster Library and is in Dean's Park.

 It houses York Minster’s library and archives as well as the Collections Department and conservation studio.

 Its name is a new one and renders homage to the part of the building that used to be the chapel of the Archbishop of York, which was built in the 13th century.

 It was refurbished in 1810 and shortly thereafter became the home of the Minster library.

 Notable items held in the collection include cathedral records dating to back to 1150 and a copy of the 1631 Wicked Bible.

 It is a Grade I listed building.

 An extension was added in 1998.
</item 5>

<item 6>
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
</item 6>

<item 7>
York Minster Police
York Minster Police is a small, specialised cathedral constabulary responsible for policing York Minster in York, United Kingdom.
</item 7>

<item 8>
Brentwood Cathedral
The Cathedral of St Mary and St Helen is the Roman Catholic cathedral in the English town of Brentwood, Essex.

 It is the seat of the Diocese of Brentwood.
</item 8>

<item 9>
Grays Court, York
Grays Court is a Grade I listed ancient house in the middle of York, England, near York Minster, behind the Treasurer's House next to the city walls.

 Grays Court is possibly the oldest continuously occupied house in the United Kingdom.

 Dating back in part to 1080 and commissioned by the first Norman Archbishop of York to provide the official residence for the Treasurers of York Minster, the
house has a significant history.

 The house was surrendered to the Crown on 26 May 1547 and the last of the mediaeval Treasurers, William Clyff, resigned.

 The first post-Reformation owner was Edward Seymour, Duke of Somerset.

 He was given the house in 1547 by King Edward VI, the son of King Henry VIII.
</item 9>

<item 10>
York Minster astronomical clock
The York Minster astronomical clock was installed in the North Transept of York Minster in 1955.

 It was first conceived in 1944 and designed by R d'E Atkinson, chief assistant at the Royal Greenwich Observatory.

 The clock is a memorial to the airmen operating from bases in Yorkshire, Durham, and Northumberland who were killed in action during World War II.
</item 10>
=== END ITEM id=b8e537162c5ce2f9 ===

=== ITEM id=2d1b272118d78c08 ===
Gold sentences and facts:
Sentence 0:  Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest commercial success is the erotic thriller "Chloe" (2009).
  Fact 0.0: Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter".
  Fact 0.1: "The Sweet Hereafter" was released in 1997.
  Fact 0.2: Egoyan received two Academy Award nominations for "The Sweet Hereafter".
  Fact 0.3: Egoyan's biggest commercial success is the erotic thriller "Chloe".
  Fact 0.4: "Chloe" was released in 2009.
Sentence 1: Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".
  Fact 1.0: Chloe is a 2009 erotic thriller film.
  Fact 1.1: Chloe was directed by Atom Egoyan.
  Fact 1.2: Chloe is a remake of the 2003 French film "Nathalie...".

Retrieved items:
<item 1>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 1>

<item 2>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 2>

<item 3>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 3>

<item 4>
Graph query row:
person: Atom Egoyan
Chloe: Chloe
ChloeConnections: ['CREATED_BY']
relatedWorks: [{'genre': 'erotic thriller', 'kind': 'film', 'published_on': '2009', 'relationship': 'CREATED_BY', 'successConnection': None,
'successRelationship': None, 'work': 'Chloe'}]
</item 4>

<item 5>
Baltasar Kormákur
Baltasar Kormákur Samper (born 27 February 1966) is an Icelandic actor, theater and film director, and film producer.

 He is best known for directing the films "101 Reykjavík", "Hafið", "A Little Trip to Heaven" (starring Julia Stiles and Forest Whitaker), a film based on the
book "Mýrin" "(Jar City)" by Arnaldur Indriðason, "Contraband", "2 Guns" (starring Mark Wahlberg and Denzel Washington) and "Everest".

 His father is the Spanish painter Baltasar Samper and actor Baltasar Breki Samper is his son.
</item 5>

<item 6>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 6>

<item 7>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 7>

<item 8>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 8>

<item 9>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 9>

<item 10>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 10>
=== END ITEM id=2d1b272118d78c08 ===

=== ITEM id=4cbd0190898e714e ===
Gold sentences and facts:
Sentence 0: Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil War.
  Fact 0.0: Am Rong was born in 1929.
  Fact 0.1: Am Rong died in May 1975.
  Fact 0.2: Am Rong was Cambodian.
  Fact 0.3: Am Rong was a soldier.
  Fact 0.4: Am Rong was a filmmaker.
  Fact 0.5: Am Rong acted as a spokesman on military matters for the Khmer Republic.
  Fact 0.6: Am Rong acted as a spokesman during the Cambodian Civil War.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is American.
  Fact 1.2: Ava Marie DuVernay is a director.
  Fact 1.3: Ava Marie DuVernay is a screenwriter.
  Fact 1.4: Ava Marie DuVernay is a film marketer.
  Fact 1.5: Ava Marie DuVernay is a film distributor.

Retrieved items:
<item 1>
List of accolades received by Selma (film)
"Selma" is a 2014 American historical drama film directed by Ava DuVernay, and produced by Dede Gardner, Jeremy Kleiner, Christian Colson, and Oprah Winfrey.

 The screenplay was written by Paul Webb.

 The film follows the events leading up to and during the Selma to Montgomery marches, and the resulting establishment of the Voting Rights Act of 1965 which
prohibited racial discrimination in voting in the United States.

 The film stars David Oyelowo as civil rights activist Martin Luther King Jr., Tom Wilkinson as President Lyndon B. Johnson, and Tim Roth as Governor of Alabama
George Wallace.

 Carmen Ejogo, Dylan Baker, Wendell Pierce, Common, and Winfrey feature in supporting roles.
</item 1>

<item 2>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 2>

<item 3>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 3>

<item 4>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 4>

<item 5>
A Wrinkle in Time (2018 film)
A Wrinkle in Time is an upcoming American science-fiction fantasy adventure film directed by Ava DuVernay and written by Jennifer Lee.

 It is based on the 1963 novel of the same name by Madeleine L'Engle.

 The film stars Oprah Winfrey, Reese Witherspoon, Mindy Kaling, Storm Reid, Zach Galifianakis and Chris Pine.

 Principal photography on the film began on November 2, 2016, in Los Angeles, California.
</item 5>

<item 6>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 6>

<item 7>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 7>

<item 8>
Emayatzy Corinealdi
Emayatzy Evett Corinealdi ( ; born January 14, 1980) is an American actress.

 She is best known for her leading role in the Ava DuVernay film "Middle of Nowhere" (2012).
</item 8>

<item 9>
Am Rong
Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil
War.

 Western journalists commented on the irony of his name as he gave briefings which "painted a rosy picture of the increasingly desperate situation on the
ground" during the war.
</item 9>

<item 10>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 10>
=== END ITEM id=4cbd0190898e714e ===

=== ITEM id=3d850774504a9d2f ===
Gold sentences and facts:
Sentence 0:  He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.
  Fact 0.0: He won three singles titles.
  Fact 0.1: He won 26 doubles titles.
  Fact 0.2: He won the Australian Open 1978 doubles title with Wojtek Fibak.
  Fact 0.3: He won the Australian Open 1980 title with fellow countryman Mark Edmondson.
  Fact 0.4: He won the Australian Open 1981 title with fellow countryman Mark Edmondson.
  Fact 0.5: He won the Roland Garros 1986 title with fellow countryman Mark Edmondson.
  Fact 0.6: He was a runner-up at the Australian Open 1985 with fellow countryman Mark Edmondson.
  Fact 0.7: Mark Edmondson is a fellow countryman of his.
Sentence 1:  In his career, he won nine doubles titles, but no singles titles.
  Fact 1.0: In his career, he won nine doubles titles.
  Fact 1.1: In his career, he won no singles titles.

Retrieved items:
<item 1>
1994 Athens International – Doubles
Horacio de la Peña and Jorge Lozano were the defending champions, but did not participate this year.
</item 1>

<item 2>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 2>

<item 3>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 3>

<item 4>
Rossana de los Ríos
Rossana de los Ríos (born 16 September 1975) is a retired professional tennis player from Paraguay.

 She was born in the Paraguayan capital, Asunción, and now lives in Miami, USA, with her daughter Ana Paula Neffa de los Ríos and her husband Gustavo Neffa, a
retired Paraguayan Association football player who played for Boca Juniors.

 Rossana achieved a career high singles ranking of No. 51 on 12 September 2001, and No. 52 in doubles on the Sony Ericsson WTA Tour.

 De los Ríos played doubles with Grand Slam champions such as Maria Sharapova, Svetlana Kuznetsova, Arantxa Sánchez Vicario, as well as Jelena Janković and
Dinara Safina.

 Rossana is also a former world number one Junior player, having won the 1992 Roland Garros Juniors event.

 De los Ríos participated in six editions (1994, 2001, 2002, 2003, 2009 and 2010) of the Australian Open.
</item 4>

<item 5>
1999 Copa Colsanitas – Doubles
The 1999 Copa Colsanitas – Doubles was the doubles event of the second edition of the Copa Colsanitas; a WTA Tier IV tournament and the most prestigious women's
tennis tournament held in Colombia and Hispanic America.

 Janette Husárová and Paola Suárez were the defending champions but only Suárez competed that year with Laura Montalvo.
</item 5>

<item 6>
1990 Stockholm Open – Doubles
Jorge Lozano and Todd Witsken were the defending champions, but lost in the quarterfinals this year.
</item 6>

<item 7>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 7>

<item 8>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 8>

<item 9>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 9>

<item 10>
1989 Volvo International – Doubles
Jorge Lozano and Todd Witsken were the defending champions but did not compete that year.
</item 10>
=== END ITEM id=3d850774504a9d2f ===

=== ITEM id=84ea1f5d695e54c7 ===
Gold sentences and facts:
Sentence 0: Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)
  Fact 0.0: Julia Compton Moore's dates are February 10, 1929 to April 18, 2004.
  Fact 0.1: Julia Compton Moore was the wife of Lieutenant General (Ret.).
Sentence 1:  Hal Moore, a United States Army officer.
  Fact 1.0: Hal Moore is a United States Army officer.
Sentence 2: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 2.0: Harold Gregory "Hal" Moore, Jr. lived from February 13, 1922 to February 10, 2017.
  Fact 2.1: Harold Gregory "Hal" Moore, Jr. was a United States Army lieutenant general.
  Fact 2.2: Harold Gregory "Hal" Moore, Jr. was an author.
Sentence 3:  He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point class (1945) to be promoted to brigadier general, major general, and lieutenant general.
  Fact 3.0: Hal Moore was a recipient of the Distinguished Service Cross.
  Fact 3.1: The Distinguished Service Cross is the U.S. military's second highest decoration for valor.
  Fact 3.2: Hal Moore was the first of his West Point class (1945) to be promoted to brigadier general.
  Fact 3.3: Hal Moore was the first of his West Point class (1945) to be promoted to major general.
  Fact 3.4: Hal Moore was the first of his West Point class (1945) to be promoted to lieutenant general.

Retrieved items:
<item 1>
List of current United States first spouses
In the United States, the first spouse (first lady for women, first gentleman for men) is the term used to refer to the spouse of a chief executive - that is,
of the spouse of the President of the United States (the First Lady of the United States and the First Gentleman of the United States) and the spouses of the
governors of the 50 U.S. states and U.S. territories (Guam, Northern Mariana Islands, Puerto Rico, American Samoa, and the United States Virgin Islands) and the
spouse of the mayor of the District of Columbia.

 (The spouses of many mayors are often called the "first lady" or "first gentleman" of the city as well, and the use of the terms sometimes extends even to the
spouses of college presidents).
</item 1>

<item 2>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 2>

<item 3>
Major general (United States)
In the United States Army, United States Marine Corps, and United States Air Force, major general is a two-star general-officer rank, with the pay grade of O-8.

 Major general ranks above brigadier general and below lieutenant general.

 A major general typically commands division-sized units of 10,000 to 15,000 soldiers.

 Major general is equivalent to the two-star rank of rear admiral in the United States Navy and United States Coast Guard and is the highest permanent rank
during peacetime in the uniformed services.

 (The higher ranks are temporary ranks linked to specific positions, although virtually all officers who have been promoted to those ranks are approved to
retire at their highest earned rank.)
</item 3>

<item 4>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 4>

<item 5>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 5>

<item 6>
Nadja West
Nadja Y. West (born 1961) is a United States Army lieutenant general and the 44th U.S. Army Surgeon General and Commanding General of the U.S. Army Medical
Command.

 West is the first black Army Surgeon General, and was the first black female, active-duty, major general and the first black female major general in Army
Medicine.

 West is also the first Army black female lieutenant general.

 She is the highest ranking female to have graduated from the United States Military Academy.

 She received the Army Distinguished Service Medal, the Defense Superior Service Medal, the Legion of Merit, and others.
</item 6>

<item 7>
Julia Compton Moore
Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)

 Hal Moore, a United States Army officer.

 Her efforts and complaints in the aftermath of the Battle of Ia Drang prompted the U.S. Army to set up survivor support networks and casualty notification
teams consisting of uniformed officers, which are still in use.
</item 7>

<item 8>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 8>

<item 9>
The text profiles military leaders across the U.S. and Allied services. James Winnefeld was U.S. vice chairman of the Joint Chiefs (2011–15); Matthew Ridgway
led airborne forces in WWII; Arthur Tedder and naval officers Arthur Power and Algernon Willis helped plan or execute Allied Mediterranean campaigns. Hal Moore
and Myron Diduryk served at Ia Drang. George Finch built the Air National Guard; Mark Mitchell earned the Distinguished Service Cross in Afghanistan. Other
</item 9>

<item 10>
Distinguished Service Cross (United States)
The Distinguished Service Cross is the second highest military award that can be given to a member of the United States Army (and previously, the United States
Army Air Forces and the United States Air Force), for extreme gallantry and risk of life in actual combat with an armed enemy force.

 Actions that merit the Distinguished Service Cross must be of such a high degree that they are above those required for all other U.S. combat decorations but
do not meet the criteria for the Medal of Honor.

 The Distinguished Service Cross is equivalent to the Navy Cross (Navy and Marine Corps), the Air Force Cross (Air Force), and the Coast Guard Cross (Coast
Guard).
</item 10>
=== END ITEM id=84ea1f5d695e54c7 ===
