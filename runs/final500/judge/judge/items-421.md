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

=== ITEM id=0bd0fc0675f6a175 ===
Gold sentences and facts:
Sentence 0: Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist, pianist, and primary songwriter for the Christian rock band Relient K.
  Fact 0.0: Matthew Arnold 'Matt' Thiessen was born August 12, 1980.
  Fact 0.1: Matt Thiessen is a Canadian-American musician, singer, and songwriter.
  Fact 0.2: Matt Thiessen is known for being co-founder of the Christian rock band Relient K.
  Fact 0.3: Matt Thiessen is known for being lead singer of the Christian rock band Relient K.
  Fact 0.4: Matt Thiessen is known for being guitarist of the Christian rock band Relient K.
  Fact 0.5: Matt Thiessen is known for being pianist of the Christian rock band Relient K.
  Fact 0.6: Matt Thiessen is known for being primary songwriter for the Christian rock band Relient K.
Sentence 1: George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.
  Fact 1.0: George Thorogood was born February 24, 1950.
  Fact 1.1: George Thorogood is an American musician, singer, and songwriter.
  Fact 1.2: George Thorogood is from Wilmington, Delaware.

Retrieved items:
<item 1>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 1>

<item 2>
Better Than the Rest
Better Than the Rest is the third album (mini-album) of songs by George Thorogood and the Destroyers, recorded in 1974 and released in 1979.

 At the time the songs were recorded, Thorogood was an unknown artist who was not signed to a record label.

 In 1979, after Thorogood had released his first two official albums on Rounder Records, "Better Than The Rest" was released by MCA.

 In 1986, the songs from this album were released on compact disc with the title "Nadine" using a different track order.
</item 2>

<item 3>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 3>

<item 4>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 4>

<item 5>
More George Thorogood and The Destroyers
More George Thorogood and the Destroyers is the fourth album by George Thorogood and The Destroyers, released in 1980.

 An alternate name for the album is I'm Wanted.
</item 5>

<item 6>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 6>

<item 7>
George Thorogood and the Destroyers (album)
George Thorogood and the Destroyers is the self-titled debut album by American blues rock band George Thorogood and the Destroyers, released in 1977.

 Consisting mostly of covers of blues hits, it included a medley of John Lee Hooker's "House Rent Boogie" and "One Bourbon, One Scotch, One Beer", the latter a
song written by Rudy Toombs for Amos Milburn, and later covered by Hooker.
</item 7>

<item 8>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 8>

<item 9>
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 9>

<item 10>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 10>
=== END ITEM id=0bd0fc0675f6a175 ===

=== ITEM id=a5ea8836b2c8616c ===
Gold sentences and facts:
Sentence 0:  It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style theme.
  Fact 0.0: The referenced piece plays during the opening sequence.
  Fact 0.1: Danny Elfman composed the piece in 1989.
  Fact 0.2: Danny Elfman composed the piece after Matt Groening approached him.
  Fact 0.3: Matt Groening is the series creator.
  Fact 0.4: Matt Groening requested a retro-style theme.
Sentence 1:  Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's Day Off".
  Fact 1.0: Clausen has scored or orchestrated music for more than 30 films and television shows.
  Fact 1.1: Clausen has scored or orchestrated music for 'Moonlighting'.
  Fact 1.2: Clausen has scored or orchestrated music for 'The Naked Gun'.
  Fact 1.3: Clausen has scored or orchestrated music for 'ALF'.
  Fact 1.4: Clausen has scored or orchestrated music for 'Ferris Bueller's Day Off'.

Retrieved items:
<item 1>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 1>

<item 2>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 2>

<item 3>
Diwa de Leon
Diwa de Leon is a composer, arranger and musician based in Manila, Philippines.

 He makes music mainly for television shows, films, and stage productions.

 His most notable work is his arrangement of GMA Network's "Survivor Philippines" theme music.

 His songs included in "Emir".

 He won the Cinema One Originals Best Musical Score award for his work on the film "Kolorete" and Best Music Score for "Kamera Obskura" at the 8th Cinemalaya
Independent Film Festival.

 He is also a founding member of the world music group Makiling.

 His first mainstream film project is Mamarazzi produced by Regal Films.
</item 3>

<item 4>
Angelo Badalamenti
Angelo Badalamenti (born March 22, 1937) is an American composer, best known for his work scoring films for director David Lynch, notably "Blue Velvet", the
"Twin Peaks" saga (1990–1992, 2017), "The Straight Story" and "Mulholland Drive".

 Badalamenti received the 1990 Grammy Award for Best Pop Instrumental Performance for his ""Twin Peaks" Theme", and has received a Lifetime Achievement Award
from the World Soundtrack Awards and the Henry Mancini Award from the American Society of Composers, Authors and Publishers.
</item 4>

<item 5>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 5>

<item 6>
Disney's Adventures of the Gummi Bears
The show is well-remembered for its theme music, written by Michael and Patty Silversher and creation of "gummiberry juice" which was a type of magic potion,
granting abilities, which allowed them to bounce away from their hunters.

 The theme song was performed by Joseph Williams, son of composer John Williams and one-time lead singer of Toto.
</item 6>

<item 7>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 7>

<item 8>
The Simpsons Theme
"The Simpsons" Theme", also referred to as "The Simpsons" Main Title Theme" in album releases, is the theme music of the animated television series "The
Simpsons".

 It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style
theme.

 The piece, which took 3 days, 2 hours, 48 minutes, and 19 seconds to create, has been noted by Elfman as the most popular of his career.

 The theme, as used for the opening sequence, was re-arranged during season 2, and the current arrangement by Alf Clausen was introduced at the beginning of the
third season.
</item 8>

<item 9>
Danny Elfman’s score for Tim Burton’s 1989 Batman, starring Michael Keaton and Jack Nicholson, was unusually released as a separate album; its iconic theme
later influenced the 1992 animated series and theme-park attractions. Prince’s Batman soundtrack album, released June 20, 1989, topped the Billboard chart for
six weeks and sold over 11 million copies. Other entries cover Yann Tiersen’s 2008 Tabarly soundtrack; Australian composer Bruce Rowland and
</item 9>

<item 10>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 10>
=== END ITEM id=a5ea8836b2c8616c ===

=== ITEM id=6d1a8349de47a1db ===
Gold sentences and facts:
Sentence 0: The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.
  Fact 0.0: The 2012 United States Libertarian National Convention was a convention of the Libertarian Party (LP).
  Fact 0.1: Delegates of the Libertarian Party chose the party's nominees at the 2012 United States Libertarian National Convention.
  Fact 0.2: The nominees chosen at the 2012 United States Libertarian National Convention were for President of the United States.
  Fact 0.3: The nominees chosen at the 2012 United States Libertarian National Convention were for Vice President of the United States.
  Fact 0.4: The nominees were chosen for the 2012 general election.
  Fact 0.5: The 2012 United States Libertarian National Convention was held May 2-6, 2012.
  Fact 0.6: The 2012 United States Libertarian National Convention was held in Las Vegas, Nevada.
  Fact 0.7: The 2012 United States Libertarian National Convention was held at the Red Rock Resort Spa and Casino.
Sentence 1:  Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.
  Fact 1.0: Gary Johnson is a former Governor of New Mexico.
  Fact 1.1: Gary Johnson won the presidential nomination.
  Fact 1.2: Gary Johnson won the presidential nomination on the first ballot.
Sentence 2: Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.
  Fact 2.0: Gary Earl Johnson was born on January 1, 1953.
  Fact 2.1: Gary Earl Johnson is an American businessman.
  Fact 2.2: Gary Earl Johnson is an American author.
  Fact 2.3: Gary Earl Johnson is an American politician.
Sentence 3:  He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.
  Fact 3.0: Gary Johnson was the 29th Governor of New Mexico.
  Fact 3.1: Gary Johnson was Governor of New Mexico from 1995 to 2003.
  Fact 3.2: Gary Johnson was a member of the Republican Party as Governor of New Mexico.

Retrieved items:
<item 1>
Political positions of Gary Johnson
Gary Johnson was the governor of New Mexico from 1995 to 2003 and ran for president in 2012 and 2016.

 In December 2011 he announced he would pursue the presidential nomination of the Libertarian Party for the 2012 general election.

 The Libertarian National Convention in May, 2012 chose Johnson as the party's candidate.

 In November 2014, Johnson announced he would pursue the presidential nomination of the Libertarian Party for the 2016 general election.

 Johnson has taken positions on many political issues as revealed through his public comments, his gubernatorial record, his Our America Initiative and his
campaigns to win the Republican and Libertarian nominations.
</item 1>

<item 2>
David Bergland
David Peter Bergland (born June 4, 1935) is an American politician who was the United States Libertarian Party's nominee at the 1983 Libertarian National
Convention for President of the United States in the 1984 presidential election.
</item 2>

<item 3>
Libertarian National Committee
The Libertarian National Committee (LNC) controls and manages the affairs, properties, and funds of the United States Libertarian Party.

 It is composed of the party officers, five at-large representatives elected every two years at the national convention, and a theoretical maximum of 10
regional representatives.

 The current chair, elected in June 2014, is Nicholas Sarwark.

 The Executive Director is Wes Benedict.
</item 3>

<item 4>
Libertarian National Convention
The Libertarian National Convention is held every two years by the Libertarian Party (United States) to choose members of the Libertarian National Committee
(LNC), and to conduct other party business.

 In presidential election years, the convention delegates enact a platform and nominate the Libertarian presidential and vice-presidential candidates who then
face the nominees of other parties in the November general election.
</item 4>

<item 5>
LPRadicals
LPRadicals is a caucus formed in 2006 within the United States Libertarian Party by Susan Hogarth and other party members who opposed removal of much of the
material in the party platform during the 2006 national party convention.

 The caucus lists four points as "key strategic principles in furthering the work of the Libertarian Party toward, as our platform describes it, 'a world set
free in our lifetime'."

 These are: Rights Are Utilitarian, Radical Abolitionism, Principled Populism, and No Particular Order (for removal of government policies).

 The caucus was active at the 2008 and 2010 Libertarian National Conventions.

 The radical caucus was revived and was extraordinarily active during the 2016 Libertarian National Convention.
</item 5>

<item 6>
Libertarian Party presidential primaries, 2016
The 2016 Libertarian Party presidential primaries and caucuses allowed electors to indicate non-binding preferences for the Libertarian Party's presidential
candidate.

 These differed from the Republican or Democratic presidential primaries and caucuses in that they did not appoint delegates to represent a candidate at the
party's convention to select the party's nominee for the United States presidential election.

 The party's nominee for the 2016 presidential election was chosen directly by registered delegates at the 2016 Libertarian National Convention, which ran from
May 26 to 30, 2016.

 The delegates nominated former New Mexico Governor Gary Johnson and former Massachusetts Governor Bill Weld.
</item 6>

<item 7>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 7>

<item 8>
Gary Johnson presidential campaign, 2012
The 2012 presidential campaign of Gary Johnson, the 29th Governor of New Mexico, was announced on April 21, 2011.

 He declared his candidacy for the 2012 Republican Party nomination for President of the United States.

 On December 28, 2011, Johnson withdrew his candidacy for the Republican nomination, and declared his candidacy for the 2012 presidential nomination of the
Libertarian Party.

 The 2012 Libertarian National Convention was held during the first weekend of May 2012.

 On May 5, 2012, after promoting his libertarian-oriented political positions to delegates, Johnson received the most votes at the convention and became the
official 2012 Libertarian presidential nominee.

 On November 6, 2012, Johnson received just under 1% of the popular vote in the general election, amounting to more than 1.2 million votes, more than double
what the Barr/Root ticket received in 2008.

 This was the most successful result for a third-party presidential candidacy since 2000, and the best in the Libertarian Party's history by vote number at the
time.

 (Johnson ran again in 2016 and received nearly four times his 2012 vote total.)
</item 8>

<item 9>
Gary Johnson
Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.

 He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.

 He was also the Libertarian Party nominee for President of the United States in the 2012 and 2016 elections.
</item 9>

<item 10>
2012 Libertarian National Convention
The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United
States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.

 Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.

 Retired California state court judge Jim Gray won the vice-presidential nomination, also on the first ballot.

 The convention also chose to replace most of the Libertarian National Committee party officers and members-at-large.
</item 10>
=== END ITEM id=6d1a8349de47a1db ===

=== ITEM id=301600b7b40d6838 ===
Gold sentences and facts:
Sentence 0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.
  Fact 0.0: The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held.
  Fact 0.1: The 2015 AMVCA was held on March 7, 2015.
  Fact 0.2: The 2015 AMVCA was held at the Eko Hotel and Suites.
  Fact 0.3: The Eko Hotel and Suites is in Victoria Island, Lagos, Nigeria.
Sentence 1: Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.
  Fact 1.0: Lagos initially emerged as a port city.
  Fact 1.1: Lagos originated on a collection of islands.
  Fact 1.2: The islands are contained in the present day Local Government Areas (LGAs) of Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa.
  Fact 1.3: The islands are separated by creeks.
  Fact 1.4: The creeks fringe the southwest mouth of Lagos Lagoon.
  Fact 1.5: The islands are protected from the Atlantic Ocean by barrier islands.
  Fact 1.6: The islands are protected from the Atlantic Ocean by long sand spits such as Bar Beach.
  Fact 1.7: The barrier islands and long sand spits, such as Bar Beach, stretch up to 100 km east and west of the mouth.

Retrieved items:
<item 1>
Lake Worth Lagoon
The Lake Worth Lagoon is a lagoon located in Palm Beach County, Florida.

 It runs parallel to the coast, and is separated from the Atlantic Ocean by barrier beaches, including Palm Beach Island.

 The lagoon is connected to the Atlantic Ocean by two permanent, man-made inlets.
</item 1>

<item 2>
Africa Magic
Africa Magic is a collection of Pay TV entertainment channels that focus on African programming.

 "Africa Magic", which started off as single channel of the same name, is a brand owned by M-Net and now comprises eight channels.

 The first "Africa Magic" channel was launched in July 2003 as a movie channel and over the next decade, the brand expanded to include seven more channels
comprising movies, television shows and general entertainment.

 Africa Magic currently broadcasts in more than 50 African countries.

 The channels include "Africa Magic Family", "Africa Magic World", "Africa Magic Showcase", "Africa Magic Yoruba", "Africa Magic Igbo", "Africa Magic Hausa".

 "Africa Magic Epic" and "Africa Magic Urban".

 Africa Magic is also responsible for the annual Africa Magic Viewers' Choice Awards (AMVCAs), the biggest celebration of film and television talent in Africa.
</item 2>

<item 3>
Eric Aghimien
Eric Enomamien Aghimien is a Nigerian director, producer, screenwriter and editor.

 His debut feature film, "A Mile from Home" won awards at both the 2014 Africa Magic Viewers Choice Awards and the 10th Africa Movie Academy Awards.
</item 3>

<item 4>
2015 Africa Magic Viewers Choice Awards
The 2015 Africa Magic Viewers Choice Awards (AMVCA) was held on March 7, 2015 at the Eko Hotel and Suites, Victoria Island, Lagos, Nigeria.

 IK and Vimbai were the hosts of the event.

 A new category, Best Indigenous Language (Igbo), was introduced in the 2015 awards.
</item 4>

<item 5>
Lake Worth Inlet
The Palm Beach Inlet, also known as the Lake Worth Inlet is an artificial cut through a barrier island connecting the northern part of the Lake Worth Lagoon in
Palm Beach County, Florida with the Atlantic Ocean.

 It is bordered by the town of Palm Beach on the south, and by the town of Palm Beach Shores to the north.

 The inlet is also the entrance channel for the Port of Palm Beach.

 Its coordinates are .
</item 5>

<item 6>
Lagos
Lagos initially emerged as a port city which originated on a collection of islands, which are contained in the present day Local Government Areas (LGAs) of
Lagos Island, Eti-Osa, Amuwo-Odofin and Apapa; the islands are separated by creeks, fringing the southwest mouth of Lagos Lagoon, while protected from the
Atlantic Ocean by barrier islands and long sand spits such as Bar Beach, which stretch up to 100 km east and west of the mouth.

 Due to rapid urbanization, the city expanded to the west of the lagoon to include areas in the present day Lagos Mainland, Ajeromi-Ifelodun and Surulere.

 This led to the classification of Lagos into two main areas - the Island, which was the initial city of Lagos, before it expanded into the area known as the
Mainland.

 This city area was governed directly by the Federal Government through the "Lagos City Council", until the creation of Lagos State in 1967, which led to the
splitting of Lagos city into the present day seven Local Government Areas (LGAs), and an addition of other towns (which now make up 13 LGAs) from the then
Western Region, to form the state.
</item 6>

<item 7>
British Academy Television Awards 2009
The 2009 British Academy Television Awards were held on 26 April at the Royal Festival Hall in London.

 The event was broadcast live on BBC One and was hosted by Graham Norton.

 The nominations were announced on 24 March.

 Winners in bold.
</item 7>

<item 8>
Africa Magic Viewers' Choice Awards
Africa Magic Viewers' Choice Awards (AMVCA) is an annual accolade presented by Multichoice recognizing outstanding achievement in television and film, voted on
by the general public.

 The inaugural Africa Magic Viewers’ Choice Awards ceremony was held in Lagos, Lagos State in Nigeria on 9 March 2013, and was broadcast live in more than 50
countries.

 Entries into the award ceremony are films and TV series that have been aired on Dstv Channels in the previous year.
</item 8>

<item 9>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 9>

<item 10>
2013 Africa Magic Viewers Choice Awards
The 2013 Africa Magic Viewers Choice Awards were held on 9 March 2014 and co-hosted by Big Brother Africa presenter IK Osakioduwa and StarGist’s host, Vimbai
Mutinhiri.

 Ivie Okujaiye and Olu Jacobs were honored with the TrailBlazer and Industry Merit Awards respectively.
</item 10>
=== END ITEM id=301600b7b40d6838 ===

=== ITEM id=194f0a9f7abafd46 ===
Gold sentences and facts:
Sentence 0: Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.
  Fact 0.0: Robin R. Bottin was born on April 1, 1959.
  Fact 0.1: Robin R. Bottin is an American special make-up effects creator.
Sentence 1:  Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing", with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".
  Fact 1.0: Robin R. Bottin is known for collaborations with directors John Carpenter, Paul Verhoeven and David Fincher.
  Fact 1.1: Robin R. Bottin worked with John Carpenter on “The Fog”.
  Fact 1.2: Robin R. Bottin worked with John Carpenter on “The Thing”.
  Fact 1.3: Robin R. Bottin worked with Paul Verhoeven on “RoboCop”.
  Fact 1.4: Robin R. Bottin worked with Paul Verhoeven on “Total Recall”.
  Fact 1.5: Robin R. Bottin worked with Paul Verhoeven on “Basic Instinct”.
  Fact 1.6: Robin R. Bottin worked with David Fincher on “Se7en”.
  Fact 1.7: Robin R. Bottin worked with David Fincher on “Fight Club”.
Sentence 2: David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.
  Fact 2.0: David Andrew Leo Fincher was born on August 28, 1962.
  Fact 2.1: David Andrew Leo Fincher is an American director and producer.
  Fact 2.2: David Andrew Leo Fincher is notably known for films, television series and music videos.
Sentence 3:  He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).
  Fact 3.0: David Fincher was nominated for the Academy Award for Best Director for “The Curious Case of Benjamin Button” (2008).
  Fact 3.1: David Fincher was nominated for the Academy Award for Best Director for “The Social Network” (2010).
Sentence 4:  For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
  Fact 4.0: For “The Social Network”, David Fincher won the Golden Globe Award for Best Director.
  Fact 4.1: For “The Social Network”, David Fincher won the BAFTA Award for Best Direction.

Retrieved items:
<item 1>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 1>

<item 2>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 2>

<item 3>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 3>

<item 4>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 4>

<item 5>
Meet the Robinsons
Meet the Robinsons is a 2007 American computer-animated science fiction comedy film produced by Walt Disney Animation Studios and released by Walt Disney
Pictures on March 30, 2007.

 The 47th Disney animated feature film, it was released in standard and Disney Digital 3-D versions.

 The film is loosely based on characters from the children's book "A Day with Wilbur Robinson", by William Joyce.

 The voice cast includes Jordan Fry, Wesley Singerman, Harland Williams, Tom Kenny, Steve Anderson, Laurie Metcalf, Adam West, Tom Selleck, and Angela Bassett.

 It was the first film released after John Lasseter became chief creative officer of Walt Disney Animation Studios.
</item 5>

<item 6>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 6>

<item 7>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 7>

<item 8>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 8>

<item 9>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 9>

<item 10>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 10>
=== END ITEM id=194f0a9f7abafd46 ===

=== ITEM id=8ccb2410402300e5 ===
Gold sentences and facts:
Sentence 0: Mastodon is an American heavy metal band from Atlanta, Georgia.
  Fact 0.0: Mastodon is an American heavy metal band.
  Fact 0.1: Mastodon is from Atlanta, Georgia.
Sentence 1:  Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals in studio (except Bill Kelliher).
  Fact 1.0: Mastodon was formed in 2000.
  Fact 1.1: Mastodon is composed of bassist Troy Sanders.
  Fact 1.2: Mastodon is composed of guitarist Brent Hinds.
  Fact 1.3: Mastodon is composed of guitarist Bill Kelliher.
  Fact 1.4: Mastodon is composed of drummer Brann Dailor.
  Fact 1.5: Troy Sanders, Brent Hinds, and Brann Dailor perform vocals in the studio.
  Fact 1.6: Bill Kelliher does not perform vocals in the studio.
Sentence 2:  Mastodon has released seven studio albums, as well as a number of other records.
  Fact 2.0: Mastodon has released seven studio albums.
  Fact 2.1: Mastodon has released a number of other records.
Sentence 3:  Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.
  Fact 3.0: Mastodon's second full-length release is "Leviathan".
  Fact 3.1: "Leviathan" is a concept album.
  Fact 3.2: "Leviathan" is based on the novel "Moby-Dick".
  Fact 3.3: "Moby-Dick" was written by Herman Melville.
Sentence 4: Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric Erlandson.
  Fact 4.0: Hole was an American alternative rock band.
  Fact 4.1: Hole formed in Los Angeles, California in 1989.
  Fact 4.2: Hole was formed by singer and guitarist Courtney Love.
  Fact 4.3: Hole was formed by lead guitarist Eric Erlandson.

Retrieved items:
<item 1>
Mastodon (band)
Mastodon is an American heavy metal band from Atlanta, Georgia.

 Formed in 2000, the group is composed of bassist Troy Sanders, guitarists Brent Hinds and Bill Kelliher, and drummer Brann Dailor, all of whom perform vocals
in studio (except Bill Kelliher).

 They all perform vocals at live shows.

 Their musical style features progressive concepts and unique instrumentation.

 Mastodon has released seven studio albums, as well as a number of other records.

 The band's debut album, "Remission", released in 2002, garnered significant critical acclaim for its unique sound.

 Mastodon's second full-length release, "Leviathan", is a concept album based on the novel "Moby-Dick" by Herman Melville.

 Three magazines awarded the record "Album of the Year" in 2004: "Revolver", "Kerrang!

" and "Terrorizer".
</item 1>

<item 2>
Blackdiamondskye
Blackdiamondskye was a 2010 North American concert tour by American bands Alice in Chains, Mastodon, and Deftones.

 Blackdiamondskye is a portmanteau of the three band's most recent albums: Alice in Chains' "Black Gives Way to Blue", Deftones' "Diamond Eyes", and Mastodon's
"Crack the Skye".

 The tour kicked off on September 16 in Chicago and culminated on October 16 in Paradise, Nevada.

 The tour lasted for 19 shows over the course of one month and featured shows at both outdoor amphitheatres and indoor venues.
</item 2>

<item 3>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 3>

<item 4>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 4>

<item 5>
Bullmoose
Bullmoose (aka Zach Dobbins) is a Canadian rock band formed in Elgin, Ontario in 1996.

 Three of the original members continue in the band, which today is based in Montreal.

 They include twin brothers Jeff Cowan (drums/vocals) and Seamus Cowan (bass/vocals) from Westport, Ontario and Eric Lawrance (guitar/vocals) from Delta,
Ontario, who launched the band when they attended school at Montreal.

 The band performs in the rock music genre, citing experimental influences of the late 1960s and early 1970s.
</item 5>

<item 6>
Hole (band)
Hole was an American alternative rock band that formed in Los Angeles, California in 1989 by singer and guitarist Courtney Love and lead guitarist Eric
Erlandson.

 The band had a revolving line-up of bassists and drummers, their most prolific being drummer Patty Schemel, and bassists Kristen Pfaff (d. 1994) and Melissa
Auf der Maur.
</item 6>

<item 7>
Pantera
Pantera was an American heavy metal band from Arlington, Texas.

 The group was formed in 1981 by the Abbott brothers – drummer Vinnie Paul and guitarist Dimebag Darrell – along with lead vocalist Terry Glaze.

 Bassist Rex Brown joined the band the following year, replacing Tommy D. Bradford, who was the unofficial original.

 Having started as a glam metal band, Pantera released four albums during the 1980s.

 Looking for a new and heavier sound, Pantera replaced Glaze with Phil Anselmo in late 1986 and released "Power Metal" in 1988.

 With its fifth album, 1990's "Cowboys from Hell", Pantera introduced a groove metal sound.

 Pantera's sixth album, 1992's "Vulgar Display of Power", exhibited an even heavier sound. "

Far Beyond Driven" (1994) debuted at number one on the "Billboard" 200.
</item 7>

<item 8>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 8>

<item 9>
The Studdogs
The band was signed by Orange Recordings in Los Angeles and toured all over the country, sharing bills with the Demolition Doll Rods, Bob Log III, The
Fleshtones, and Immortal Lee County Killers and a one-time show with the Suicide Girls Burlesque Tour in Orlando.

 While in the Studdogs, Rich Evans began promoting concerts under the tag "Mutiny Productions".

 He now heads the independent "Florida's Dying" label, promoting and releasing material from the Florida-underground music scene.

 Kyle Justin joined the band on drums in early 2002 to replace Jason, who was moving to Texas.

 Kyle vacated his position as drummer in 2003 to pursue a music career in Los Angeles.

 Eric Gebhardt left the band in 2005 to pursue a solo career under the alias Red Mouth.
</item 9>

<item 10>
I Was Totally Destroying It
I Was Totally Destroying It is a five-piece powerpop band from Chapel Hill, North Carolina, featuring current and former members of Saddle Creek band Sorry
About Dresden.

 After self-releasing and touring the country behind their first full-length album, the band signed with Portland-based indie label Greyday Records.

 Since forming in 2007, the band has toured much of the country, playing with artists such as Conor Oberst and the Mystic Valley Band, Motion City Soundtrack,
Cursive, Superchunk, Portastatic, Ryan Ferguson (formerly of No Knife), Dear and the Headlights, Margot & the Nuclear So and So's, Snowden, Billy Bragg,
Annuals, Joan Jett, The Actual, Ozma, Polvo, Birds of Avalon and Blankface.

 Their next full-length, Horror Vacui, was released by Greyday in October 2009, and was co-produced by Josh Cain of Motion City Soundtrack.
</item 10>
=== END ITEM id=8ccb2410402300e5 ===

=== ITEM id=d6088fee7b003fbe ===
Gold sentences and facts:
Sentence 0: Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
  Fact 0.0: Robert Paul Holdstock was born on 2 August 1948.
  Fact 0.1: Robert Paul Holdstock died on 29 November 2009.
  Fact 0.2: Robert Paul Holdstock was an English novelist.
  Fact 0.3: Robert Paul Holdstock was an English author.
  Fact 0.4: Robert Paul Holdstock was best known for his works of Celtic fantasy literature.
  Fact 0.5: Robert Paul Holdstock was best known for his works of Nordic fantasy literature.
  Fact 0.6: Robert Paul Holdstock was best known for his works of Gothic fantasy literature.
  Fact 0.7: Robert Paul Holdstock was best known for his works of Pictish fantasy literature.
  Fact 0.8: Robert Paul Holdstock's works are predominantly in the fantasy subgenre of mythic fiction.
Sentence 1: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 1.0: Henry Louis Mencken was born on September 12, 1880.
  Fact 1.1: Henry Louis Mencken died on January 29, 1956.
  Fact 1.2: Henry Louis Mencken was an American journalist.
  Fact 1.3: Henry Louis Mencken was an American satirist.
  Fact 1.4: Henry Louis Mencken was an American cultural critic.
  Fact 1.5: Henry Louis Mencken was a scholar of American English.

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
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 2>

<item 3>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
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
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
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
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 7>

<item 8>
Robert Holdstock bibliography
This is a bibliography of fantasy author Robert Holdstock.
</item 8>

<item 9>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 9>

<item 10>
Litbits
Litbits is a literary discussion radio programme and podcast, hosted by academic Adam Smyth and literary journalist James Kidd.

 Guests on the programme have included Polly Russell, Paul Myerscough, Alex von Tunzelmann, Samantha Ellis, Jonathan Beckman, Hallie Rubenhold, Steve Rose, Joe
Brooker, Dennis Duncan, Dinah Roe, Giles Milton and Liane Strauss.

 Subjects covered include Rooms and Procrastination.
</item 10>
=== END ITEM id=d6088fee7b003fbe ===

=== ITEM id=023c97f5a108df9f ===
Gold sentences and facts:
Sentence 0: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 0.0: California State Polytechnic University, Pomona is also called CPP, Cal Poly Pomona, or Cal Poly.
  Fact 0.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 0.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.
Sentence 1:  It is one of two polytechnics in the California State University system.
  Fact 1.0: California State Polytechnic University, Pomona is one of two polytechnics in the California State University system.
Sentence 2:  Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.
  Fact 2.0: The university is named after the first President of Russia, B.N. Yeltsin.
  Fact 2.1: The full name of the university is 'Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina'.
  Fact 2.2: The university, Ural Federal University, is often shortened to UrFU or УрФУ.
  Fact 2.3: Ural Federal University was formed by a merger of the Ural State Technical University and Ural State University.
  Fact 2.4: Ural Federal University is one of the leading educational institutions in the Ural region.
Sentence 3:  Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.
  Fact 3.0: Ural Federal University acts as a research and innovation center of the Ural region.
  Fact 3.1: Ural Federal University has close cooperation with the Russian Academy of Sciences.

Retrieved items:
<item 1>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 1>

<item 2>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 2>

<item 3>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 3>

<item 4>
Ural Federal University
The Ural Federal University named after the first President of Russia B. N. Yeltsin (Russian: Уральский федеральный университет имени первого Президента России
Б.Н.

 Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural
State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.

 Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.

 Training of students is carried out in four main areas of knowledge and 108 academic majors.

 The links between the university and intermediate educational institutions are actively developed.
</item 4>

<item 5>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 5>

<item 6>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 6>

<item 7>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 7>

<item 8>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 8>

<item 9>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 9>

<item 10>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 10>
=== END ITEM id=023c97f5a108df9f ===

=== ITEM id=98cc0b9220a49e55 ===
Gold sentences and facts:
Sentence 0: Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.
  Fact 0.0: Charles Hastings Judd was a Colonel.
  Fact 0.1: Charles Hastings Judd was born on September 8, 1835.
  Fact 0.2: Charles Hastings Judd died on April 18, 1890.
  Fact 0.3: Charles Hastings Judd was an American businessman.
  Fact 0.4: Charles Hastings Judd was an American rancher.
  Fact 0.5: Charles Hastings Judd was an American courtier.
  Fact 0.6: Charles Hastings Judd was an American politician of the Kingdom of Hawaii.
  Fact 0.7: Charles Hastings Judd was a businessman, rancher, courtier and politician of the Kingdom of Hawaii.
Sentence 1:  He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
  Fact 1.0: Charles Hastings Judd served as Chamberlain to King Kalakaua.
  Fact 1.1: Charles Hastings Judd served as Colonel of the military staff of King Kalakaua.
  Fact 1.2: Charles Hastings Judd traveled with King Kalakaua on his 1881 world tour.
Sentence 2: Kalākaua (November 16, 1836 – January 20, 1891), born David Laʻ amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalākaua and sometimes called The Merrie Monarch, was the last king (and second to last monarch) of the Kingdom of Hawaiʻ i.
  Fact 2.0: Kalakaua was born on November 16, 1836.
  Fact 2.1: Kalakaua died on January 20, 1891.
  Fact 2.2: Kalakaua was born David La'amea Kamananakapu Mahinulani Naloiaehuokalani Lumialani Kalakaua.
  Fact 2.3: Kalakaua is sometimes called The Merrie Monarch.
  Fact 2.4: Kalakaua was the last king of the Kingdom of Hawaii.
  Fact 2.5: Kalakaua was the second-to-last monarch of the Kingdom of Hawaii.

Retrieved items:
<item 1>
Corps of Military Staff Clerks
The Corps of Military Staff Clerks was an administrative corps of the Canadian Army.

 The Headquarters Corps of Military Staff Clerks was established in 1905.
</item 1>

<item 2>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 2>

<item 3>
Algernon Willis
Admiral of the Fleet Sir Algernon Usborne Willis (17 May 1889 – 12 April 1976) was a Royal Navy officer.

 He served in the First World War and saw action at the Battle of Jutland in May 1916.

 He also served in the Second World War as Commander-in-Chief, South Atlantic in which capacity he led actions against German and Japanese raiding ships.

 He continued his war service as Flag Officer commanding 3rd Battle Squadron and Second in command of the Eastern Fleet and then as Flag Officer commanding
Force H, the force which covered North African Operations, the Allied invasion of Sicily in July 1943 and then the Allied invasion of Italy in September 1943.

 He spent the final years of the war as Commander-in-Chief, Levant, in which capacity he conducted naval operations in support of the Dodecanese Campaign, and
then as Second Sea Lord, in which capacity he arranged the manpower for the campaign in the Pacific Ocean against the Imperial Japanese Navy.

 After the war he served as Commander-in-Chief, Mediterranean Fleet, in which role he was faced with unrest in Mandatory Palestine, before he became
Commander-in-Chief, Portsmouth.
</item 3>

<item 4>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 4>

<item 5>
Arthur Tedder, 1st Baron Tedder
Marshal of the Royal Air Force Arthur William Tedder, 1st Baron Tedder, {'1': ", '2': ", '3': ", '4': "} (11 July 1890 – 3 June 1967) was a senior Royal Air
Force commander.

 He was a pilot and squadron commander in the Royal Flying Corps in the First World War and he went on to serve as a senior officer in the Royal Air Force
during the inter-war years when he served in Turkey, Great Britain and the Far East.

 During the Second World War, as Air Officer Commanding RAF Middle East Command, Tedder directed air operations in the Mediterranean and North Africa, including
the evacuation of Crete and "Operation Crusader" in North Africa.

 His bombing tactics became known as the "Tedder Carpet".

 Later in the war Tedder took command of Mediterranean Air Command and in that role was closely involved in the planning of the Allied invasion of Sicily and
then the Allied invasion of Italy.

 When Operation Overlord—the invasion of France—came to be planned, Tedder was appointed Deputy Supreme Commander at Supreme Headquarters Allied Expeditionary
Force under General Eisenhower.
</item 5>

<item 6>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 6>

<item 7>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 7>

<item 8>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 8>

<item 9>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
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
=== END ITEM id=98cc0b9220a49e55 ===

=== ITEM id=74e895dada6cda62 ===
Gold sentences and facts:
Sentence 0: Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.
  Fact 0.0: Scott Richard Weiland was born Scott Richard Kline.
  Fact 0.1: Scott Richard Weiland was born on October 27, 1967.
  Fact 0.2: Scott Richard Weiland's date range ends on December 3, 2015.
  Fact 0.3: Scott Richard Weiland was an American musician.
  Fact 0.4: Scott Richard Weiland was an American singer.
  Fact 0.5: Scott Richard Weiland was an American songwriter.
Sentence 1: Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.
  Fact 1.0: Richard Willis Hawley was born on 17 January 1967.
  Fact 1.1: Richard Willis Hawley is English.
  Fact 1.2: Richard Willis Hawley is a guitarist.
  Fact 1.3: Richard Willis Hawley is a singer-songwriter.
  Fact 1.4: Richard Willis Hawley is a producer.

Retrieved items:
<item 1>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 1>

<item 2>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 2>

<item 3>
Hugh Whitaker
Hugh Whitaker (born 18 May 1961) is an English musician and the former drummer for the British indie rock band The Housemartins.

 He replaced original drummer Chris Lang and drummed for the band's first album, "London 0 Hull 4", and its attendant single releases.

 He left the band before the recording of their second album, "The People Who Grinned Themselves to Death".

 Whitaker left the band on amicable terms and even participated in the promotional video for the band's first single without him, "Five Get Over Excited",
wherein he was kidnapped by his replacement, Dave Hemingway, and locked in a hessian sack.
</item 3>

<item 4>
Bill Scott (author)
William Neville "Bill" Scott OAM (4 October 1923 – 22 December 2005) was an Australian author, folklorist, songwriter, poet and a collector of bush ballads and
Australian folk history.

 He has published anthologies of Australian bush songs, including the best selling book "The Complete Book of Australian Folklore" published in 1976.

 He was awarded the Order of Australia in 1992 for his contributions to folklore, folk music and Australian literature.

 He was considered a living treasure and his anthologies of songs and his donated collections continue his legacy.
</item 4>

<item 5>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 5>

<item 6>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 6>

<item 7>
A Compilation of Scott Weiland Cover Songs
A Compilation of Scott Weiland Cover Songs is a compilation album and third solo studio album by Stone Temple Pilots frontman Scott Weiland, released on August
30, 2011.

 The album consists entirely of cover songs, featuring an array of covers of artists that inspired Weiland, such as David Bowie, The Beatles, The Rolling
Stones, and The Smiths.

 The album was originally set to be released together with Weiland's memoir "Not Dead & Not for Sale" but Weiland decided later to release the album by itself
exclusively in digital format.
</item 7>

<item 8>
Standing at the Sky's Edge
Standing at the Sky's Edge is the seventh studio album from English musician Richard Hawley, released in the UK on 7 May 2012 and in the US on 28 August 2012.

 The album is markedly different from Hawley's previous efforts, often relinquishing softer instrumentation in favour of squalling guitars.

 As with all of Hawley's previous albums, the title obliquely refers to a location in his native city of Sheffield, in this case Skye Edge, a hillside area with
views over the city centre and formerly known for its crime-ridden estates but largely redeveloped in the mid-2000s.
</item 8>

<item 9>
Richard Hawley
Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.

 After his first band Treebound Story (formed while he was still at school) broke up, Hawley found success as a member of Britpop band Longpigs in the 1990s.

 After that group broke up in 2000, he later joined the band Pulp, led by his friend Jarvis Cocker, for a short time.

 As a solo musician, Hawley has released seven studio albums.

 He has been nominated for a Mercury prize twice and once for a Brit Award.

 He has collaborated with Lisa Marie Presley, Arctic Monkeys, Manic Street Preachers and Paul Weller.
</item 9>

<item 10>
John Scott Whiteley
John Scott Whiteley (born 1950) is an English organist and composer.

 He has performed extensively around the world and since 1985 has undertaken an annual tour of the USA.

 He has performed in most major UK Cathedrals and concert halls, and was Assistant Organist and later Organist and Director of the Girls' choir at York Minster
between 1975 and 2010.

 He is currently Organist Emeritus of York Minster.
</item 10>
=== END ITEM id=74e895dada6cda62 ===
