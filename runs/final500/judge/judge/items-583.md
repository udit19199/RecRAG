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

=== ITEM id=f8008f9afeb629bb ===
Gold sentences and facts:
Sentence 0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who successfully worked as the film director, the screenwriter, the art director and the animator.
  Fact 0.0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич) was born on 4 September 1924.
  Fact 0.1: Vladimir Petrovich Danilevich died on 9 October 2001.
  Fact 0.2: Vladimir Petrovich Danilevich was a well-known Soviet animator.
  Fact 0.3: Vladimir Petrovich Danilevich was a well-known Russian animator.
  Fact 0.4: Vladimir Petrovich Danilevich successfully worked as a film director.
  Fact 0.5: Vladimir Petrovich Danilevich successfully worked as a screenwriter.
  Fact 0.6: Vladimir Petrovich Danilevich successfully worked as an art director.
  Fact 0.7: Vladimir Petrovich Danilevich successfully worked as an animator.
Sentence 1: Wolfgang Becker (born 22 June 1954) is a German film director and writer.
  Fact 1.0: Wolfgang Becker was born on 22 June 1954.
  Fact 1.1: Wolfgang Becker is a German film director.
  Fact 1.2: Wolfgang Becker is a German writer.
Sentence 2:  He is best known to the international audience for his work "Good Bye Lenin!
  Fact 2.0: Wolfgang Becker is best known to the international audience for his work "Good Bye Lenin!".
Sentence 3: " (2003).
  Fact 3.0: "Good Bye Lenin!" is from 2003.

Retrieved items:
<item 1>
Graph query row:
entity: Wolfgang Becker
relationship: CREATED_BY
related: Good Bye Lenin!
</item 1>

<item 2>
Good Bye Lenin! (soundtrack)
Good Bye Lenin!

 is the original soundtrack album of the film of the same title starring Daniel Brühl and Katrin Sass.
</item 2>

<item 3>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 3>

<item 4>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 4>

<item 5>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 5>

<item 6>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 6>

<item 7>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 7>

<item 8>
Good Bye, Lenin!
Good Bye, Lenin!

 is a 2003 German tragicomedy film, directed by Wolfgang Becker.

 The cast includes Daniel Brühl, Katrin Saß, Chulpan Khamatova, and Maria Simon.

 Most scenes were shot at the Karl-Marx-Allee in Berlin and around Plattenbauten near Alexanderplatz.
</item 8>

<item 9>
Stefan Ruzowitzky
Stefan Ruzowitzky (born 25 December 1961) is an Austrian film director and screenwriter.
</item 9>

<item 10>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 10>
=== END ITEM id=f8008f9afeb629bb ===

=== ITEM id=bd0ab87b57f1736f ===
Gold sentences and facts:
Sentence 0: Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.
  Fact 0.0: Talking Heads were an American rock band.
  Fact 0.1: Talking Heads were formed in 1975.
  Fact 0.2: Talking Heads were formed in New York City.
  Fact 0.3: Talking Heads were active until 1991.
Sentence 1:  is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
  Fact 1.0: The group is a surf rock group.
  Fact 1.1: The group formed in Auburn, Alabama.
  Fact 1.2: The group formed in the early 1990s.
  Fact 1.3: The group came to prominence over the following decade.

Retrieved items:
<item 1>
New World Man
"New World Man" is a track from the 1982 album "Signals" by Canadian rock band Rush.

 The song was the last and quickest composed song on the album, stemming from a suggestion by then-Rush producer Terry Brown to even out the lengths of the two
sides of the cassette version.

 It went to #1 (on the "RPM" national singles chart) in Canada, where it remained for two weeks in October 1982.

 Less successful in the United States, it nonetheless remains Rush's only American Top 40 hit, peaking at #21 on the "Billboard" singles chart for three weeks
in October and November 1982.

 It also topped the "Billboard" Top Tracks chart for two weeks (their first single to do so).

 "New World Man" also reached #42 in the UK; a remixed version released as a double A-side with "Countdown" later reached #36 in the UK in early 1983.
</item 1>

<item 2>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 2>

<item 3>
No Talking, Just Head
No Talking, Just Head is an album released in 1996 by The Heads, a band composed of Jerry Harrison, Tina Weymouth, and Chris Frantz of Talking Heads, joined by
a variety of guest singers.

 Its name may be seen as an allusion to the fact that Talking Heads' former vocalist, David Byrne, is the only member not involved.
</item 3>

<item 4>
Talking Heads Africa
Talking Heads (Africa) was introduced in Cape Town in 2008 as part of the Infecting the City public art festival.

 Talking Heads has four core components that form the project.

 These include: developing a platform for conversation and exchange with and between experts; creating a network of African thought leaders; shooting
mini-documentaries that define these leaders and their contributions; developing the tools to make this model work in cities all over the African continent.
</item 4>

<item 5>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 5>

<item 6>
Man or Astro-man?
Man or Astro-man?

 is a surf rock group that formed in Auburn, Alabama, in the early 1990s and came to prominence over the following decade.
</item 6>

<item 7>
Manimal Vinyl
Manimal has offices in Los Angeles and New York City.

 In 2015, Manimal signed on with legendary avant-garde artist Yoko Ono to release the highly anticipated sequel to 2007's "Yes, I'm A Witch" with "Yes, I'm A
Witch Too" which features collaborations and remixes from Death Cab for Cutie, Miike Snow, Penguin Prison, Moby, Peter Bjorn and John, Cibo Matto, and Sparks.
</item 7>

<item 8>
Talking Heads
Talking Heads were an American rock band formed in 1975 in New York City and active until 1991.

 The band comprised David Byrne (lead vocals, guitar), Chris Frantz (drums), Tina Weymouth (bass), and Jerry Harrison (keyboards, guitar).

 Described by critic Stephen Thomas Erlewine as "one of the most critically acclaimed bands of the '80s," the group helped to pioneer new wave music by
integrating elements of punk, art rock, funk, and world music with avant-garde sensibilities and an anxious, clean-cut image.
</item 8>

<item 9>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 9>

<item 10>
More Songs About Buildings and Food
More Songs About Buildings and Food is the second studio album by the American rock band Talking Heads, released in July 1978.

 The album was the first of three Talking Heads LPs produced by collaborator Brian Eno.

 It saw the group move musically toward a danceable style, crossing singer David Byrne's unusual delivery with new emphasis on the rhythm section (made up of
bassist Tina Weymouth and drummer Chris Frantz).
</item 10>
=== END ITEM id=bd0ab87b57f1736f ===

=== ITEM id=1831cc8cd77afca2 ===
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
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
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
John Eric Holmes
John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine,
an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and
science fiction.

 His father, Wilfred "Jasper" Holmes was also a writer of adventure stories under the pen name Alec Hudson.

 His writings appeared under his full name and under variants such as Eric Holmes and J. Eric Holmes and the pen name Sidney Leland.
</item 5>

<item 6>
Graph query row:
person: Hal Moore
occupation: ['United States Army lieutenant general', 'author']
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
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
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
=== END ITEM id=1831cc8cd77afca2 ===

=== ITEM id=d357526c874e9fa9 ===
Gold sentences and facts:
Sentence 0:  The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic Conference (MAAC).
  Fact 0.0: The Gaels were led by sixth year head coach Tim Cluess.
  Fact 0.1: The Gaels played their home games at the Hynes Athletic Center.
  Fact 0.2: The Gaels were members of the Metro Atlantic Athletic Conference (MAAC).
Sentence 1:  The Cyclones were coached by Steve Prohm, who was in his 1st season.
  Fact 1.0: The Cyclones were coached by Steve Prohm.
  Fact 1.1: Steve Prohm was in his 1st season.

Retrieved items:
<item 1>
Tim Esmay
Tim Esmay is an American baseball coach.

 He is the former head coach of the Arizona State Sun Devils baseball team.

 He was the head coach of the Arizona State Sun Devils baseball team from prior to the 2010 season until the end of the 2014 season, when he announced his
resignation.

 Esmay is an Arizona State alumnus, and he played baseball there from 1986–1987.

 After graduating from Arizona State in 1987, Esmay served as an assistant at Arizona State, Grand Canyon, and Utah.

 He was Utah's head coach from 1997–2004, before he was hired as an assistant at Arizona State prior to the 2005 season.
</item 1>

<item 2>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 2>

<item 3>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 3>

<item 4>
Rudy Lavik
Rudolph H. "Rudy" Lavik (April 30, 1892 – September 29, 1979) was an American football and basketball coach and college athletics administrator.

 He served as the head football coach at Concordia College in Moorhead, Minnesota from 1920 to 1921, at Arizona State Teacher's College of Flagstaff—now
Northern Arizona University—from 1928 to 1932, and at Arizona State Teachers College at Tempe—now Arizona State University—from 1933 to 1937, compiling a career
college football record of 37–42–7.

 Lavik was also the head basketball coach at Arizona State Flagstaff (1927–1931) and Arizona State Tempe (1933–1935, 1939–1948), tallying a career college
basketball mark of 146–137.

 In addition he served as the athletic director at Arizona State from 1933 to 1949.

 He was a graduate of Springfield College in Springfield, Massachusetts.
</item 4>

<item 5>
2015–16 Iona Gaels men's basketball team
The 2015–16 Iona Gaels men's basketball team represented Iona College during the 2015–16 NCAA Division I men's basketball season.

 The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference (MAAC).

 They finished the season 22–11, 16–4 in MAAC play to finish in second place.

 They defeated Canisius, Siena, and Monmouth to be champions of the MAAC Tournament and earn the conference's automatic bid to the NCAA Tournament where, as a
#13 seed, they lost in the first round to Iowa State.
</item 5>

<item 6>
Bob Hurley
Robert Emmet Hurley, Sr. (born July 31, 1947) is a basketball coach at St. Anthony High School in Jersey City, New Jersey.

 Hurley has amassed 28 state championships and more than 1000 wins in 39 years as a coach.

 On February 2, 2011, Hurley became the tenth coach in high school history to win 1000 games.

 He is featured in the documentary "The Street Stops Here".

 Five of his teams have gone undefeated.

 On April 5, 2010, he was announced as the only coach to be inducted into the Naismith Memorial Basketball Hall of Fame that year and only the third high school
coach in history to be so honored; he was formally inducted on August 13 of that year.

 Hurley is the father of Bobby Hurley, a former All-American point guard at Duke and the head basketball coach at Arizona State, and Dan Hurley, who was hired
in February 2012 to coach the University of Rhode Island after two years of coaching at Wagner College and nine years coaching at Newark's Saint Benedict's
Preparatory School, also one of the top high school programs in the nation.
</item 6>

<item 7>
2012–13 Iona Gaels men's basketball team
The 2012–13 Iona Gaels men's basketball team represented Iona College during the 2012–13 NCAA Division I men's basketball season.

 The Gaels, led by third year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 20–14, 11–7 in MAAC play to finish in a tie for fourth place.

 They were champions of the MAAC Tournament, defeating Manhattan in the championship game, to earn an automatic bid to the 2013 NCAA Tournament where they lost
in the second round to Ohio State.
</item 7>

<item 8>
2012–13 Murray State Racers men's basketball team
The 2012–13 Murray State Racers men's basketball team represented Murray State University during the 2012–13 NCAA Division I men's basketball season.

 The Racers, led by second year head coach Steve Prohm, played their home games at the CFSB Center and were members of the West Division of the Ohio Valley
Conference.

 They finished the season 21–10, 10–6 in OVC play to be champions of the West Division.

 They advanced to the championship game of the OVC Tournament where they lost to Belmont in overtime.

 Despite the 21 wins, they did not participate in a post season tournament.
</item 8>

<item 9>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 9>

<item 10>
Dan Hurley
Daniel S. Hurley (born January 16, 1973) is currently the head basketball coach at the University of Rhode Island.

 He was named head coach on March 20, 2012 after a two-year stint at Wagner College.

 Prior to Wagner, Hurley had a standout career as head coach of Saint Benedict's Preparatory School, where he built the New Jersey school into one of the top
high school basketball programs in America.

 Dan Hurley is the son of Hall of Fame high school coach Bob Hurley and younger brother of former Duke and Sacramento Kings guard Bobby Hurley.
</item 10>
=== END ITEM id=d357526c874e9fa9 ===

=== ITEM id=9f17872fe5884d3d ===
Gold sentences and facts:
Sentence 0: Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.
  Fact 0.0: Bluebeard's Castle is a one-act opera.
  Fact 0.1: Bluebeard's Castle is by Hungarian composer Béla Bartók.
  Fact 0.2: The Hungarian title of Bluebeard's Castle is A kékszakállú herceg vára.
  Fact 0.3: Literally, the Hungarian title means The Blue-Bearded Duke's Castle.
Sentence 1: Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.
  Fact 1.0: Agrippina (HWV 6) is an opera seria.
  Fact 1.1: Agrippina (HWV 6) has three acts.
  Fact 1.2: Agrippina (HWV 6) is by George Frideric Handel.
  Fact 1.3: Agrippina (HWV 6) has a libretto by Cardinal Vincenzo Grimani.

Retrieved items:
<item 1>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 1>

<item 2>
One Act Play (band)
One Act Play is an instrumental rock, post-rock band from Dallas, Texas.

 Heavily influenced by and often compared to bands such as Mogwai, Explosions In The Sky, and The Cancer Conspiracy, One Act Play has been known for their
impassioned and diverse sound, from simple profound elements of post rock to off-the-wall psychedelic/progressive patterns.
</item 2>

<item 3>
Suor Angelica
Suor Angelica (Sister Angelica) is an opera in one act by Giacomo Puccini to an original Italian libretto by Giovacchino Forzano.

 It is the second opera of the trio of operas known as "Il trittico" ("Triptych").

 It received its world premiere at the Metropolitan Opera on December 14, 1918.
</item 3>

<item 4>
Uncle Samuel
Uncle Samuel is a one-act comic opera with a libretto by Arthur Law and music by George Grossmith.

 It was first produced at the Opera Comique on 3 May 1881 to 8 October 1881, as companion piece to "Patience".

 The piece also toured from December 1887 to June 1888 as a companion piece to "H.M.S. Pinafore".
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
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 6>

<item 7>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 7>

<item 8>
Das geheime Königreich
Das geheime Königreich ("The Secret Kingdom") is an opera in one act with words and music by Ernst Krenek, his Op.

 50 and the second of three one-acters (with "Der Diktator" and "Schwergewicht, oder Die Ehre der Nation") which premiered on 6 May 1928 at the Hessisches
Staatstheater Wiesbaden as part of the Mai-Festwoche Wiesbaden.

 It is subtitled "Märchenoper in 1 Akt " (fairy-tale opera in one act) and has been called a satirical fairy-tale opera.
</item 8>

<item 9>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 9>

<item 10>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 10>
=== END ITEM id=9f17872fe5884d3d ===

=== ITEM id=a75521454e8cb6c5 ===
Gold sentences and facts:
Sentence 0: Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct "Final Destination 5" six years later, and filmed in the IMAX 3D format.
  Fact 0.0: Aliens of the Deep is a 2005 documentary film.
  Fact 0.1: Aliens of the Deep was directed in part by James Cameron.
  Fact 0.2: James Cameron directed Aliens of the Deep alongside fellow cameraman and friend Steven Quale.
  Fact 0.3: Steven Quale would go on to direct Final Destination 5 six years later.
  Fact 0.4: Aliens of the Deep was filmed in the IMAX 3D format.
Sentence 1:  Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of the planet's more unusual forms of life.
  Fact 1.0: James Cameron teams with NASA scientists to explore the Mid-Ocean Ridges.
  Fact 1.1: The Mid-Ocean Ridges are submerged chains of mountains.
  Fact 1.2: The Mid-Ocean Ridges are in the Atlantic and Pacific oceans.
  Fact 1.3: The Mid-Ocean Ridges are home to some of the planet's more unusual forms of life.
Sentence 2: Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.
  Fact 2.0: Poetry in Motion is a 1982 Canadian documentary film.
  Fact 2.1: Poetry in Motion was directed by Ron Mann.
  Fact 2.2: Poetry in Motion features contemporary North American poetry.
  Fact 2.3: Poetry in Motion features contemporary North American music.
Sentence 3:  Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.
  Fact 3.0: Poetry in Motion features some Black Mountain poets.
  Fact 3.1: Poetry in Motion features some Beat poets.
  Fact 3.2: Poetry in Motion features some minimalist poets.
  Fact 3.3: Poetry in Motion features some avant-garde poets.
Sentence 4:  It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.
  Fact 4.0: Poetry in Motion was released in theaters.
  Fact 4.1: Poetry in Motion was later distributed on VHS.
  Fact 4.2: Poetry in Motion was later distributed on LaserDisc.
  Fact 4.3: Poetry in Motion was later distributed on DVD.
Sentence 5:  An extended CD-ROM version was also released.
  Fact 5.0: An extended CD-ROM version of Poetry in Motion was also released.

Retrieved items:
<item 1>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 1>

<item 2>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 2>

<item 3>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 3>

<item 4>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 4>

<item 5>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 5>

<item 6>
War/Dance
War/Dance is a 2007 American documentary film written and directed by Sean Fine and Andrea Nix Fine and produced by Shine Global's Susan MacLaury, a professor
at Kean University, and Albie Hecht.

 It was nominated for the 2008 Academy Award for Best Documentary Feature and received the Emmy Award for Best Documentary and Best Cinematography in 2010.
</item 6>

<item 7>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 7>

<item 8>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 8>

<item 9>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 9>

<item 10>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 10>
=== END ITEM id=a75521454e8cb6c5 ===

=== ITEM id=e1461b355f078cc0 ===
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
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 1>

<item 2>
Astrobiology Magazine
Astrobiology Magazine (exploring the solar system and beyond), or Astrobiology Mag, is an American NASA-sponsored international online popular science magazine
containing popular science content, which refers to articles for the general reader on science and technology subjects.

 The magazine reports on missions of NASA and other space agencies, as well as presents news of relevant research conducted by various institutions,
universities, and non-profit groups.

 In addition, the magazine provides a forum through which researchers and the general public can oversee the progress made in fields of study that are
associated with the science of astrobiology.

 According to Phys.org, the magazine has a "vast archive of stories covering a broad array of topics ... [and] covers science and nature topics relevant to
space, innovation and biology, with an emphasis on the existence, detection and exploration of life in the universe".

 The magazine was created by Helen Matsos, presently Chief Editor and Executive Producer, and began publication in 2000.
</item 2>

<item 3>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 3>

<item 4>
Rap-Up
Rap-Up is a magazine launched in 2001 by founder Devin Lazerine.

 The publication was originally a website devoted to hip hop, until Lazerine decided to pitch the possibility of a magazine to several publishers.

 The magazine is focused on the hip hop and R&B aspect of the music industry, and predominantly features interviews with artists, actors and other entertainers.

 Issued quarterly, the magazine's target audience was the 14 to 28 demographic, although the demographic has become older over time.
</item 4>

<item 5>
Lorin Thompson
Lorin Thompson, Lorin Hartwell Jr Thompson, (1911–1997), was a muralist, artist, and creator and illustrator of the character "Ranger Rick" for the National
Wildlife Federation's children's magazine, Ranger Rick.
</item 5>

<item 6>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
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
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 9>

<item 10>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 10>
=== END ITEM id=e1461b355f078cc0 ===

=== ITEM id=b97e5fbf51b7960c ===
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
Reagan Democrat
A Reagan Democrat is a traditionally Democratic voter in the United States, referring especially to white working-class Rust Belt residents, who defected from
their party to support Republican President Ronald Reagan in either or both of the 1980 and 1984 elections as well as Republican Presidents George H. W. Bush in
the 1988 election and George W. Bush in either or both of the 2000 and 2004 elections.

 Part of this group also defected to Donald Trump, the Republican candidate in 2016, who won in Pennsylvania, Ohio, Michigan, Wisconsin, and Iowa, all states
that voted for Reagan but went Democratic in 2008 and 2012.
</item 1>

<item 2>
Elliott Abrams
Elliott Abrams (born January 24, 1948) is a former American diplomat, lawyer and political scientist who served in foreign policy positions for Presidents
Ronald Reagan and George W. Bush.

 Abrams was convicted of withholding information from Congress about the Iran–Contra affair while serving under Reagan, but was pardoned by President George H.
W. Bush.
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
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
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
=== END ITEM id=b97e5fbf51b7960c ===

=== ITEM id=29aaaf48a2864ebf ===
Gold sentences and facts:
Sentence 0: Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.
  Fact 0.0: Terri Kathleen Nunn was born June 26, 1961.
  Fact 0.1: Terri Kathleen Nunn is an American singer.
  Fact 0.2: Terri Kathleen Nunn is an American actress.
Sentence 1: Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.0: Sonya Scarlet was born 2 April 1980.
  Fact 1.1: Sonya Scarlet is the singer of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.2: Sonya Scarlet is the lyricist of the Italian extreme gothic metal band Theatres des Vampires.

Retrieved items:
<item 1>
Terri Nunn
Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.

 She is best known as the lead vocalist of the new wave/synthpop band Berlin.
</item 1>

<item 2>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 2>

<item 3>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 3>

<item 4>
CeCe Peniston’s 1998 “Nobody Else,” intended to lead a canceled album, never appeared on its planned compilation; she later released two Silk singles. Cherie
Currie’s 2015 *Reverie* ended a 35-year album gap and was Kim Fowley’s final production; her son finished it after Fowley died. *Young and Wild* compiles Currie
sisters’ recordings and a new song. Their 1980 *Messin’ with the
</item 4>

<item 5>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 5>

<item 6>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 6>

<item 7>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 7>

<item 8>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 8>

<item 9>
Sonya Yoncheva
Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
</item 9>

<item 10>
Sonya Scarlet
Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
</item 10>
=== END ITEM id=29aaaf48a2864ebf ===

=== ITEM id=9b9c0d70cbded256 ===
Gold sentences and facts:
Sentence 0: Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on the German island of Greifswalder Oie in December 1937.
  Fact 0.0: Operation Lighthouse was the name given to a failed experimental launch.
  Fact 0.1: The Operation Lighthouse launch involved four Aggregate 3 liquid-fuel rockets.
  Fact 0.2: Wernher von Braun conducted the Operation Lighthouse launch.
  Fact 0.3: Walter Dornberger conducted the Operation Lighthouse launch.
  Fact 0.4: The Operation Lighthouse launch took place on the German island of Greifswalder Oie.
  Fact 0.5: The Operation Lighthouse launch took place in December 1937.
Sentence 1: Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War II.
  Fact 1.0: Major-General Dr. Walter Robert Dornberger was born on 6 September 1895.
  Fact 1.1: Walter Robert Dornberger died on 27 June 1980.
  Fact 1.2: Walter Robert Dornberger was a German Army artillery officer.
  Fact 1.3: Walter Robert Dornberger's career spanned World War I.
  Fact 1.4: Walter Robert Dornberger's career spanned World War II.

Retrieved items:
<item 1>
Magnus von Braun
Magnus "Mac" Freiherr von Braun (10 May 1919 – 21 June 2003) was a German chemical engineer, Luftwaffe aviator, and rocket scientist at Peenemünde, the
Mittelwerk, and after emigrating to the United States via Operation Paperclip, at Fort Bliss.

 He was the brother of Sigismund and Wernher von Braun.
</item 1>

<item 2>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 2>

<item 3>
Von Braun Ferry Rocket
Von Braun Ferry Rocket was a concept design for a shuttle spacecraft that was developed by Wernher von Braun in a seminal series of early-1950s Collier's
magazine articles, Man Will Conquer Space Soon!

 by Wernher von Braun "et al." The Ferry Rocket concept has evolved over time.
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 5>

<item 6>
Von Braun (crater)
von Braun, named after the rocket pioneer Wernher von Braun, is a lunar crater located near the northwestern limb of the Moon.

 It lies along the western edge of the Oceanus Procellarum, to the northeast of the crater Lavoisier.

 The northeastern rim of this crater is on the edge of the Sinus Roris, a bay feature in the northwestern part of the Oceanus Procellarum.

 Due to its proximity to the limb, this crater appears significantly foreshortened when viewed from the Earth.
</item 6>

<item 7>
Man Will Conquer Space Soon!
Man Will Conquer Space Soon!

 was the title of a famous series of 1950s magazine articles in "Collier's" detailing Wernher von Braun's plans for manned spaceflight.

 Edited by Cornelius Ryan, the individual articles were authored by such space notables of the time as Willy Ley, Fred Lawrence Whipple, Dr. Joseph Kaplan, Dr.
Heinz Haber, and von Braun.

 The articles were illustrated with paintings and drawings by Chesley Bonestell, Fred Freeman, and Rolf Klep, some of the finest magazine illustrators of the
time.
</item 7>

<item 8>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 8>

<item 9>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 9>

<item 10>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 10>
=== END ITEM id=9b9c0d70cbded256 ===

=== ITEM id=b794b26409d08fad ===
Gold sentences and facts:
Sentence 0: The Great War is a 26-episode documentary series from 1964 on the First World War.
  Fact 0.0: The Great War is a documentary series.
  Fact 0.1: The Great War has 26 episodes.
  Fact 0.2: The Great War is a documentary series from 1964.
  Fact 0.3: The Great War is about the First World War.
Sentence 1:  The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian Broadcasting Commission.
  Fact 1.0: The Great War was a co-production of the Imperial War Museum.
  Fact 1.1: The Great War was a co-production of the British Broadcasting Corporation.
  Fact 1.2: The Great War was a co-production of the Canadian Broadcasting Corporation.
  Fact 1.3: The Great War was a co-production of the Australian Broadcasting Commission.
Sentence 2:  The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.
  Fact 2.0: Michael Redgrave was the narrator of The Great War.
  Fact 2.1: Marius Goring gave readings in The Great War.
  Fact 2.2: Ralph Richardson gave readings in The Great War.
  Fact 2.3: Cyril Luckham gave readings in The Great War.
  Fact 2.4: Sebastian Shaw gave readings in The Great War.
  Fact 2.5: Emlyn Williams gave readings in The Great War.
Sentence 3: Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.
  Fact 3.0: Imperial War Museums (IWM) is a British national museum organisation.
  Fact 3.1: Imperial War Museums has branches at five locations in England.
  Fact 3.2: Three of the five Imperial War Museums branches are in London.

Retrieved items:
<item 1>
Norman Gilliland
Norman Gilliland has been a producer on Wisconsin Public Radio since 1984, where he hosts classical music broadcasts, produces the interview program "University
of the Air," and reads for "Chapter A Day."

 He holds degrees in English and Broadcasting from the University of Florida and attended graduate school in English at Duke University, where he developed an
interest in broadcasting.

 He is also an active author with four published books, the historical novel "Sand Mansions" and its stand-alone sequel "Midnight Catch," "Downeast Ledge"
(2013), plus two nonfiction books about classical music--"Grace Notes for a Year" and "Scores to Settle."

 He has produced an audio drama based upon Dick Ringler's modern English translation of the Old English narrative Beowulf titled "Beowulf: The Complete Story—A
Drama" (ISBN  ).

 He was one of a handful of experts interviewed in the Academy Award winning short documentary .
</item 1>

<item 2>
Michael Palin: Around the World in 80 Days
Around the World in 80 Days is a 7-part BBC television travel series first broadcast on BBC1 in 1989.

 It was presented by comedian and actor Michael Palin.

 The show was inspired by Jules Verne's classic novel "Around the World in Eighty Days", in which a character named Phileas Fogg accepts a wager to
circumnavigate the globe in eighty days or less.

 Palin was given the same deadline, and not allowed to use aircraft, which did not exist in Jules Verne's time and would make completing the journey far too
easy.

 He followed Phileas Fogg's route as closely as possible.

 Along the way he commented on the sights and cultures he encountered.

 Palin encountered several setbacks during his voyage, partly because he travelled with a five-person film crew, who are collectively named after Passepartout,
Phileas Fogg's manservant.
</item 2>

<item 3>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 3>

<item 4>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 4>

<item 5>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 5>

<item 6>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 6>

<item 7>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 7>

<item 8>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 8>

<item 9>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 9>

<item 10>
First Life (TV series)
First Life is a 2010 British nature documentary series written and presented by David Attenborough, also known by the expanded titles David Attenborough's First
Life (UK) and First Life with David Attenborough (USA).

 It was first broadcast in the USA as a two-hour special on the Discovery Channel on 24 October 2010.

 In the United Kingdom it was broadcast as a two-part series on BBC Two on 5 November 2010.

 "First Life" sees Attenborough tackle the subject of the origin of life on Earth.

 He investigates the evidence from the earliest fossils, which suggest that complex animals first appeared in the oceans around 540 million years ago, an event
known as the Cambrian Explosion.

 Trace fossils of multicellular organisms from an even earlier period, the Ediacaran biota, are also examined.

 Attenborough travels to Canada, Morocco and Australia, using some of the latest fossil discoveries and their nearest equivalents amongst living species to
reveal what life may have been like at that time.

 Visual effects and computer animation are used to reconstruct and animate the extinct life forms.
</item 10>
=== END ITEM id=b794b26409d08fad ===

=== ITEM id=16598008fc51ae85 ===
Gold sentences and facts:
Sentence 0: Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
  Fact 0.0: Wilmington International Airport has the IATA code ILM.
  Fact 0.1: Wilmington International Airport has the ICAO code KILM.
  Fact 0.2: Wilmington International Airport has the FAA LID ILM.
  Fact 0.3: Wilmington International Airport is a public airport.
  Fact 0.4: Wilmington International Airport is located just north of Wilmington, North Carolina.
  Fact 0.5: Wilmington International Airport is located in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
Sentence 1: Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km) northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.
  Fact 1.0: Texarkana Regional Airport has the IATA code TXK.
  Fact 1.1: Texarkana Regional Airport has the ICAO code KTXK.
  Fact 1.2: Texarkana Regional Airport has the FAA LID TXK.
  Fact 1.3: Texarkana Regional Airport is also known as Webb Field.
  Fact 1.4: Texarkana Regional Airport is a public use airport.
  Fact 1.5: Texarkana Regional Airport is located three nautical miles (6 km) northeast of the central business district of Texarkana.
  Fact 1.6: Texarkana is a city in Miller County, Arkansas, United States.

Retrieved items:
<item 1>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 1>

<item 2>
Pensacola International Airport
Pensacola International Airport (IATA: PNS, ICAO: KPNS, FAA LID: PNS) , formerly Pensacola Gulf Coast Regional Airport and Pensacola Regional Airport (Hagler
Field), is a public use airport three nautical miles (6 km) northeast of the central business district of Pensacola, in Escambia County, Florida, United States.

 It is owned by the City of Pensacola.

 Despite the name, this airport does not offer direct international flights.

 This airport is one of the five major airports in North Florida, others being: Northwest Florida Beaches International Airport, Destin-Fort Walton Beach
Airport Tallahassee International Airport, and Jacksonville International Airport.
</item 2>

<item 3>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 3>

<item 4>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 4>

<item 5>
Graph query row:
airport: Texarkana Regional Airport
geographic_connection: Texarkana
relationships: ['LOCATED_IN']
</item 5>

<item 6>
Wilmington International Airport
Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated
Wrightsboro, Cape Fear Township, New Hanover County.
</item 6>

<item 7>
Graph query row:
airport: Wilmington International Airport
geographic_connection: Wrightsboro
relationships: ['LOCATED_IN']
</item 7>

<item 8>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 8>

<item 9>
Elizabeth City Regional Airport
Elizabeth City Regional Airport (IATA: ECG, ICAO: KECG, FAA LID: ECG) is a joint civil-military public and military use airport located three nautical miles
(6 km) southeast of the central business district of Elizabeth City, in Pasquotank County, North Carolina, United States.

 The airport, on the shore of the Pasquotank River, is also known as Elizabeth City-Pasquotank County Regional Airport or ECG Regional Airport.

 It is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "general aviation" facility.
</item 9>

<item 10>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 10>
=== END ITEM id=16598008fc51ae85 ===
