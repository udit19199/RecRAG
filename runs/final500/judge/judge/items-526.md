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

=== ITEM id=2ec493ec4f9e60c0 ===
Gold sentences and facts:
Sentence 0: The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.
  Fact 0.0: The Shame of a City is a 2006 feature-length documentary.
  Fact 0.1: The Shame of a City premiered at the Philadelphia Film Festival.
Sentence 1: Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and preserved in the WPA Slave Narrative Collection.
  Fact 1.0: Unchained Memories is a 2003 documentary film.
  Fact 1.1: Unchained Memories is about the stories of former slaves.
  Fact 1.2: The former slaves were interviewed during the 1930s.
  Fact 1.3: The interviews were part of the Federal Writers' Project.
  Fact 1.4: The stories are preserved in the WPA Slave Narrative Collection.

Retrieved items:
<item 1>
The Sorrow and the Pity
The Sorrow and the Pity (French: Le Chagrin et la Pitié ) is a two-part 1969 documentary film by Marcel Ophüls about the collaboration between the Vichy
government and Nazi Germany during World War II.

 The film uses interviews with a German officer, collaborators, and resistance fighters from Clermont-Ferrand.

 They comment on the nature of and reasons for collaboration.

 The reasons include antisemitism, anglophobia, fear of Bolsheviks and Soviet invasion, the desire for power, and simple caution.
</item 1>

<item 2>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 2>

<item 3>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 3>

<item 4>
Unchained Memories
Unchained Memories is a 2003 documentary film about the stories of former slaves interviewed during the 1930s as part of the Federal Writers' Project and
preserved in the WPA Slave Narrative Collection.

 This HBO film interpretation directed by Ed Bell and Thomas Lennon is a compilation of slave narratives, narrated by actors, emulating the original
conversation with the interviewer.

 The slave narratives may be the most accurate in terms of the everyday activities of the enslaved, serving as personal memoirs of more than two thousand former
slaves.

 The documentary depicts the emotions of the slaves and what they endured.

 The "Master" had the opportunity to sell, trade, or kill the enslaved, for retribution should one slave not obey.
</item 4>

<item 5>
The Shame of a City
The Shame of a City is a 2006 feature-length documentary, which premiered at the Philadelphia Film Festival.

 Filmmaker Tigre Hill chronicles the 2003 Philadelphia mayoral race between Democrat incumbent mayor John Street and Republican challenger Sam Katz.

 Early polls showed Katz with a small lead in this predominantly Democratic city but twenty-seven days before the election, an FBI bug was found in the mayor’s
office.

 The discovery at first seemed like a death knell to the Street campaign and a near certain victory for Katz.

 Yet this prediction was proven wrong when Street and his supporters successfully polarized the campaign by leveling accusations of instituational racial
prejudice and playing on historical skepticism of the Republican-controlled federal government.

 As a result, Street won re-election by a sixteen-point margin.
</item 5>

<item 6>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 6>

<item 7>
Chicago: City on the Make
Chicago: City on the Make is an essay by Nelson Algren published in 1951.

 Initially greeted with scorn by critics and newspaper editors in the city of its gaze (The "Chicago Daily News" famously called it a "Case for Ra(n)t
Control"), it is now widely regarded by scholars as the definitive prose portrait of the city of Chicago, although it has never rivaled the literary status of
Carl Sandburg's 1916 poem "Chicago."

 Algren leans heavily on the imagery and themes developed by Sandburg, to whom Algren dedicated the book.

 Curiously, he also quietly leans upon a poem about New York called "The City" by Ben Maddow, from whom Algren lifted powerful images of urban life.

 Subsequent portraits of Chicago, such as Studs Terkel's 1985 "Chicago", have likewise leaned heavily upon Algren's work.
</item 7>

<item 8>
Naked City (TV series)
Naked City is a police drama series from Screen Gems which was broadcast from 1958 to 1959 and from 1960 to 1963 on the ABC television network.

 It was inspired by the 1948 motion picture "The Naked City" and mimics its dramatic "semi-documentary" format.

 As in the film, each episode concluded with a narrator intoning the iconic line: "There are eight million stories in the naked city.

 This has been one of them."
</item 8>

<item 9>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 9>

<item 10>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 10>
=== END ITEM id=2ec493ec4f9e60c0 ===

=== ITEM id=562dd730b9683ee5 ===
Gold sentences and facts:
Sentence 0: Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an SS-"Obergruppenführer".
  Fact 0.0: Karl Hermann Frank's life dates are 24 January 1898 to 22 May 1946.
  Fact 0.1: Karl Hermann Frank was a prominent Sudeten German Nazi official in Czechoslovakia.
  Fact 0.2: Karl Hermann Frank was a Nazi official in Czechoslovakia prior to World War II.
  Fact 0.3: Karl Hermann Frank was a Nazi official in Czechoslovakia during World War II.
  Fact 0.4: Karl Hermann Frank was an SS-Obergruppenführer.
Sentence 1:  Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".
  Fact 1.0: Until April 1942, the Obergruppenführer rank was the highest commissioned SS rank.
  Fact 1.1: Until April 1942, the Obergruppenführer rank was inferior only to Reichsführer-SS.
  Fact 1.2: Reichsführer-SS was the SS rank held by Heinrich Himmler.
  Fact 1.3: RFSS was the internal SS abbreviation for Himmler.
  Fact 1.4: Obergruppenführer translates as senior group leader.
  Fact 1.5: The rank of Obergruppenführer was senior to Gruppenführer.

Retrieved items:
<item 1>
Obergruppenführer
Obergruppenführer (] , "senior group leader") was a Nazi Party paramilitary rank that was first created in 1932 as a rank of the "Sturmabteilung" (SA), and
adopted by the "Schutzstaffel" (SS) one year later.

 Until April 1942, it was the highest commissioned SS rank, inferior only to "Reichsführer-SS" (Heinrich Himmler or RFSS, which was the internal SS-abbreviation
for Himmler) Translated as "senior group leader", the rank of "Obergruppenführer" was senior to "Gruppenführer".

 A similarly named rank of "Untergruppenführer" existed in the SA from 1929 to 1930 and as a title until 1933.

 In April 1942, the new rank of "SS-Oberst-Gruppenführer" was created which was above "Obergruppenführer" and below "Reichsführer-SS".
</item 1>

<item 2>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 2>

<item 3>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 3>

<item 4>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
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
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 6>

<item 7>
Karl Hermann Frank
Karl Hermann Frank (24 January 1898 – 22 May 1946) was a prominent Sudeten German Nazi official in Czechoslovakia prior to and during World War II and an
SS-"Obergruppenführer".

 He was tried, convicted and executed after World War II for his role in organizing the massacres of the people of the Czech villages of Lidice and Ležáky.
</item 7>

<item 8>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 8>

<item 9>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 9>

<item 10>
Rudolf Plajner
Dr. Rudolf Plajner (April 5, 1901 in Prostějov, Moravia – June 23, 1987) was declared the Chief Scout of the newly unified Czech Scouts and Guides association
called Junák, after its inception on January 22, 1939.

 Junák was abolished by force and Scouting prohibited by German State Secretary Karl Hermann Frank during the Nazi occupation of Czechoslovakia on October 28,
1940.

 After World War II, the association was reborn.

 His function was confirmed at the third Junák council or meeting (cs:sněm) in 1968, and served in this function until the end of his life.
</item 10>
=== END ITEM id=562dd730b9683ee5 ===

=== ITEM id=7eba28b5c0d34696 ===
Gold sentences and facts:
Sentence 0: Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the electronica band Republica.
  Fact 0.0: Samantha Marie Sprackling was born on 3 June 1968.
  Fact 0.1: Samantha Marie Sprackling was born in Lagos, Nigeria.
  Fact 0.2: Samantha Marie Sprackling is more commonly known as Saffron.
  Fact 0.3: Samantha Marie Sprackling is a singer.
  Fact 0.4: Samantha Marie Sprackling is a songwriter.
  Fact 0.5: Samantha Marie Sprackling is an actress.
  Fact 0.6: Samantha Marie Sprackling is the lead singer of the electronica band Republica.
Sentence 1: Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.
  Fact 1.0: Myles Richard Kennedy was born Myles Richard Bass.
  Fact 1.1: Myles Richard Kennedy was born on November 27, 1969.
  Fact 1.2: Myles Richard Kennedy is an American musician.
  Fact 1.3: Myles Richard Kennedy is an American singer.
  Fact 1.4: Myles Richard Kennedy is an American songwriter.
  Fact 1.5: Myles Richard Kennedy is best known as the lead vocalist and rhythm guitarist of the rock band Alter Bridge.
  Fact 1.6: Myles Richard Kennedy is the lead vocalist in guitarist Slash's backing band.
  Fact 1.7: Slash's backing band is known as Myles Kennedy and the Conspirators.
Sentence 2:  A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several artists, and has been involved with several projects throughout his career.
  Fact 2.0: Myles Richard Kennedy is a former guitar instructor.
  Fact 2.1: Myles Richard Kennedy is from Spokane, Washington.
  Fact 2.2: Myles Richard Kennedy has worked as a session musician.
  Fact 2.3: Myles Richard Kennedy has worked as a songwriter.
  Fact 2.4: Myles Richard Kennedy has made studio appearances with several artists.
  Fact 2.5: Myles Richard Kennedy has made live appearances with several artists.
  Fact 2.6: Myles Richard Kennedy has been involved with several projects throughout his career.

Retrieved items:
<item 1>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 1>

<item 2>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 2>

<item 3>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 3>

<item 4>
List of songs recorded by Myles Kennedy
Myles Kennedy is an American rock musician.

 Born in Boston, Massachusetts, he originally began his musical career in Spokane, Washington as the guitarist in jazz band Cosmic Dust.

 After recording 1991's "Journey", he left to form alternative rock band Citizen Swing in 1992.

 The group released two albums, 1992's "Cure Me with the Groove" and 1996's "Deep Down", on both of which Kennedy was co-credited for songwriting.

 Kennedy's next band was The Mayfield Four, which he co-founded with guitarist Craig Johnson, bassist Marty Meisner and drummer Zia Uddin in 1996.

 The frontman wrote most of the material on the band's 1998 debut "Fallout" and all of the songs on 2001's "Second Skin".
</item 4>

<item 5>
Myles Kennedy discography
The full discography of rock musician Myles Kennedy consists of eleven studio albums, two concert films, four live albums, two extended plays, and thirteen
singles in total, in addition to eleven studio tracks that he has appeared on as a featured artist, one of which was a single.

 Born in Boston on November 27, 1969, Kennedy is currently a member of the rock band Alter Bridge, with whom he has released four studio albums, two concert
films, and several singles.

 He is also the frontman of Slash's touring group, and with Slash he has released a live album, "Live in Manchester", the first of a series of live albums
released throughout the summer of 2010, and "", another live album released in 2011.

 In 2012, he released a collaboration studio album with Slash titled "Apocalyptic Love", which is billed to Slash featuring Myles Kennedy and the Conspirators,
as well as the 2014 followup titled "World on Fire".

 With The Mayfield Four, he released two studio albums, two extended plays, and four singles; with Citizen Swing, two studio albums; and with Cosmic Dust, one
studio album.
</item 5>

<item 6>
Myles Kennedy
Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm
guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.

 A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several
artists, and has been involved with several projects throughout his career.
</item 6>

<item 7>
Bent to Fly
"Bent to Fly" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it was released as the second single from the guitarist's third solo album (the second with Kennedy and The Conspirators), "World
on Fire".

 The song was used as the theme song for the 2014 National Rugby League Finals series, and Slash performed the song live at ANZ Stadium as pre-show
entertainment for the league's grand final that year.
</item 7>

<item 8>
World on Fire (Slash song)
"World on Fire" is a song by American hard rock guitarist Slash, featuring vocalist Myles Kennedy and backing band The Conspirators.

 Written by Slash and Kennedy, it is the title track of the guitarist's third solo album (the second with Kennedy and The Conspirators) "World on Fire".

 Released as the album's lead single, "World on Fire" topped the US "Billboard" Mainstream Rock Songs chart.
</item 8>

<item 9>
Graph query row:
person: Myles Kennedy
occupation: ['rock musician', 'guitarist', 'frontman']
associated_with: Cosmic Dust
role: None
</item 9>

<item 10>
Apocalyptic Love World Tour
The Apocalyptic Love World Tour is the second concert tour by ex-Guns N' Roses lead guitarist Slash as a solo artist.

 The tour features the same backing band that toured with Slash during his first solo tour, though this time the act is billed as "Slash featuring Myles Kennedy
and The Conspirators".

 The tour is in support of Slash's second solo album "Apocalyptic Love", which features Myles Kennedy handling lead vocal duties on all songs, as well as his
touring band members: bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 10>
=== END ITEM id=7eba28b5c0d34696 ===

=== ITEM id=062d8066de6bdd85 ===
Gold sentences and facts:
Sentence 0:  It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!
  Fact 0.0: The entity referred to as 'It' is known for producing the long-running series 'Tom Goes to the Mayor'.
  Fact 0.1: The entity referred to as 'It' is known for producing the long-running series 'Nathan For You'.
  Fact 0.2: The entity referred to as 'It' is known for producing the long-running series 'Tim and Eric Awesome Show, Great Job!'.
Sentence 1: " and "Check It Out!
  Fact 1.0: 'Check It Out!' is a series title.
Sentence 2:  with Dr. Steve Brule".
  Fact 2.0: The series title is 'Check It Out! with Dr. Steve Brule'.
Sentence 3: Check It Out!
  Fact 3.0: Check It Out! is a title.
Sentence 4:  with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!
  Fact 4.0: Check It Out! with Dr. Steve Brule is an American sketch comedy television series.
  Fact 4.1: Check It Out! with Dr. Steve Brule is a spin-off of 'Tim and Eric Awesome Show, Great Job!'.

Retrieved items:
<item 1>
Kate Berlant
Kate Berlant (born July 16, 1987) is an American stand-up comedian, actress, and writer.

 She is known for appearing on "The Meltdown with Jonah and Kumail" and starring in an episode of the Netflix original series "".

 With John Early, Berlant created the Vimeo original series "555" produced by Abso Lutely Productions.

 Her father is the artist Tony Berlant.
</item 1>

<item 2>
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 2>

<item 3>
Bagboy (TV special)
Bagboy is a 2015 American television special produced for Adult Swim, and aired on February 21, 2015, to positive critical reception.

 Written and directed by Tim Heidecker and Eric Wareheim for Abso Lutely Productions and John C. Reilly, the special is a fictional sitcom pilot in the universe
of "Tim and Eric Awesome Show, Great Job!

", written by and starring Reilly's recurring "Tim and Eric" character Dr. Steve Brule.

 This is the second "Tim and Eric" spin-off to feature the Dr. Steve Brule character after "Check it Out!

 with Dr. Steve Brule".
</item 3>

<item 4>
Abso Lutely Productions
Abso Lutely Productions is a film and television production company owned by actor and producers Tim Heidecker, Eric Wareheim and producer Dave Kneebone.

 It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!

" and "Check It Out!

 with Dr. Steve Brule".
</item 4>

<item 5>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 5>

<item 6>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 6>

<item 7>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 7>

<item 8>
Upright Citizens Brigade (TV series)
Upright Citizens Brigade is an American sketch comedy television series that premiered on August 19, 1998 on Comedy Central.

 The show aired for three seasons with each season consisting of ten episodes.

 The series featured four members of Upright Citizens Brigade, an improvisational sketch comedy group.

 The cast included Matt Besser, Amy Poehler, Ian Roberts, and Matt Walsh.

 The cast would later reunite for another series of a similar format that premiered in 2016 on Seeso.
</item 8>

<item 9>
Hot Package
Hot Package is an Adult Swim entertainment variety show, created by Derrick Beckles.

 The show parodies network entertainment shows such as "Entertainment Tonight" and "Access Hollywood".

 Instead of sourcing its news from real celebrities, TV shows, and films, all of Hot Package's "entertainment" news comes from found footage, including clips
from forgotten B Films and bizarre TV shows.

 The show is hosted by Derrick Beckles, Pat O'Brien, Anastasia Roark, and Mark McGrath, and features colorful guests, makeovers, and interview segments.

 "Hot Package", produced by Abso Lutely Productions, Abominable Pictures, TV Carnage, and Williams Street, premiered on October 4, 2013, and has currently aired
eleven episodes.

 On May 9, 2014, Adult Swim confirmed that Hot Package would be returning for a second season.
</item 9>

<item 10>
Check It Out! with Dr. Steve Brule
Check It Out!

 with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!

" starring John C. Reilly as Dr. Steve Brule.

 The series premiered on Cartoon Network's late night programming block, Adult Swim, on May 16, 2010.

 The program follows Brule as he examines different facets of living.

 His severe naivete and social awkwardness generally land him in embarrassing situations, though he largely remains ignorant of any embarrassment he's causing
himself.

 As the series progresses, he reveals shocking and sometimes horrifying details about his past and personal life.
</item 10>
=== END ITEM id=062d8066de6bdd85 ===

=== ITEM id=fa8bacacfd7caddb ===
Gold sentences and facts:
Sentence 0:  She was born in Guangzhou, China.
  Fact 0.0: The woman referred to as "she" was born in Guangzhou, China.
Sentence 1:  These videos are an overview and recap of each year's viral videos, events, memes, and music.
  Fact 1.0: The videos being described are an overview of each year's viral videos, events, memes, and music.
  Fact 1.1: The videos being described are a recap of each year's viral videos, events, memes, and music.

Retrieved items:
<item 1>
The entries profile filmmakers Pamela B. Green, Ava DuVernay, Howard Barish, Dana Brown, Matthew O’Neill, and the Maya Deren Award; actors Emayatzy Corinealdi
and Issa Rae; musicians Emel Mathlouthi, Siti binti Saad, and Angélique Kidjo; fitness figures Ingrid Romero and Anais Zanotti; and YouTubers Lilly Singh, Wayne
Goss, and Wengie. They
</item 1>

<item 2>
Issa Rae (born Jo-Issa Rae Diop on January 12, 1985) is an American actress, writer, director, producer, and web series creator. Best known for creating
YouTube’s *Awkward Black Girl*, she later launched a platform showcasing content by people of color. Her shows have earned more than 20 million views and
260,000 subscribers.
</item 2>

<item 3>
CVX Live
CVX Live (Creator Viewer Experience Live) is an event designed to bring online media creators and their fans together.

 It was held for the first time in Orem, Utah, on August 7 to 8, 2015 Originally created by the team behind the "Bored Shorts TV" YouTube channel, the
convention is Utah's first YouTube event, gathering thousands of online YouTube creators and fans together in one place.

 One of the creators, Brett Robers, described it as “...a very unique opportunity to bring YouTube to life.”
</item 3>

<item 4>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 4>

<item 5>
YouTube Rewind
YouTube Rewind is a video series produced and created by YouTube and Portal A Interactive.

 These videos are an overview and recap of each year's viral videos, events, memes, and music.

 Each year, the number of YouTube celebrities featured in the video, as well as the presentation of the series, have increased.

 The latest episode of "YouTube Rewind" was released on December 8, 2016.
</item 5>

<item 6>
Lilly Singh
Lilly Singh (born 26 September 1988) is a Canadian YouTube personality, vlogger, comedian, writer, and actress.

 She is better known by her YouTube username IISuperwomanII.

 Since beginning her channel in October 2010, her videos have received over 2 billion views, and her channel has accumulated over 12 million subscribers.

 In 2016, she was ranked 3rd on the "Forbes" list of the world's highest paid YouTube stars (behind Roman Atwood and PewDiePie), earning a reported $7.5 million
in 2016.

 Singh has featured in the annual YouTube Rewind every year since 2014.

 She ranked 1st on 2017 Forbes Top Influencers List in the entertainment category.
</item 6>

<item 7>
Wayne Goss (make-up artist)
Wayne Alan Goss (born 4 March 1978) is an English makeup artist, YouTube personality and entrepreneur.

 Best known for his "Wayne Goss" channel on YouTube, he is the creator and owner of the self-eponymous makeup brush line launched in September 2013.

 As of November 2016, Goss' main YouTube channel had over 2.7 million subscribers and had had 344 million views.
</item 7>

<item 8>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 8>

<item 9>
Kevin Nalty
Kevin 'Nalts' Nalty (born May 12, 1969) is a YouTube comedian and partner based in Doylestown, Pennsylvania better known under his YouTube username Nalts.

 Nalts began on YouTube as one of the top-20 most-viewed comedy channels, and collaborates with many of today's top YouTube personalities.

 He has more than 1,000 videos which, as of April 18, 2016, have been viewed more than 297 million times on YouTube alone.

 He has been ranked as one of YouTube's Most Subscribed users.

 He is the author of "Beyond Viral: How to Attract Customers, Promote Your Brand, and Make Money with Online Video (Wiley & Sons, 2010).
</item 9>

<item 10>
Wengie
Wendy Ayche (born January 9 1986) known professionally as Wengie, is an Australian YouTube personality and vlogger.

 She was born in Guangzhou, China.

 She got the nickname "Wengie" at a dance class when two people gave her a nickname inspired by her Chinese name, Wén Jié.

 Since starting her channel on February 11 2013, her videos have received over 281 million views, and her channel has accumulated over 11 followers In the
January 2016 edition of "ElleGirl Japan", she was nominated as a channel to watch.

 In August 2016, she was ranked the 5th Fastest growing channel in the world.

 Ayche was featured in the annual YouTube Rewind in 2016 and was also singled out by Google as one of the top beauty creators in the Asia region.

 Ayche reached 5 million YouTube subscribers in January 2017, and her channel is currently in the 12th most subscribed How-To & Style Channel on YouTube.

 She is also currently a board member of the Internet Creators Guild, a non-profit focused on providing the protection, representation and guidance to online
creators.
</item 10>
=== END ITEM id=fa8bacacfd7caddb ===

=== ITEM id=b116b150acd7a527 ===
Gold sentences and facts:
Sentence 0: George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
  Fact 0.0: George Albert Smith Sr. was born on April 4, 1870.
  Fact 0.1: George Albert Smith Sr. died on April 4, 1951.
  Fact 0.2: George Albert Smith Sr. was an American religious leader.
  Fact 0.3: George Albert Smith Sr. served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
Sentence 1: Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.
  Fact 1.0: Wong Jing was born on 3 May 1955.
  Fact 1.1: Wong Jing is a Hong Kong film director.
  Fact 1.2: Wong Jing is a Hong Kong film producer.
  Fact 1.3: Wong Jing is a Hong Kong actor.
  Fact 1.4: Wong Jing is a Hong Kong presenter.
  Fact 1.5: Wong Jing is a Hong Kong screenwriter.

Retrieved items:
<item 1>
D. Arthur Haycock
David Arthur Haycock (September 4, 1916 – February 25, 1994) was a personal secretary to several twentieth-century presidents of The Church of Jesus Christ of
Latter-day Saints (LDS Church), including George Albert Smith, Joseph Fielding Smith, Harold B. Lee, Spencer W. Kimball and Ezra Taft Benson.
</item 1>

<item 2>
George A. Smith
George Albert Smith (June 26, 1817 – September 1, 1875) (known throughout his life as George A. Smith) was an early leader in the Latter-day Saint movement.

 He served in the Quorum of the Twelve Apostles and as a member of the First Presidency of The Church of Jesus Christ of Latter-day Saints (LDS Church).
</item 2>

<item 3>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 3>

<item 4>
Justin Smith (milliner)
Justin Smith (born 1978) is a leading British milliner based in London.

 He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.

 Smith's hats have been exhibited around the world, and have been acquired by such museums as the Fashion Institute of Technology in New York and the Victoria &
Albert Museum in London for Hats: An Anthology curated by Stephen Jones.

 His work is in continuous demand by stylists and photographers for the quality style press.

 He has been visiting lecturer at London's Royal College of Art and has conducted seminars and teaching workshops for the British Council.
</item 4>

<item 5>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 5>

<item 6>
William Farrar Smith
William Farrar Smith (February 17, 1824 – February 28, 1903), known as ‘Baldy’ Smith, was a Union general in the American Civil War, notable for attracting the
extremes of glory and blame.

 He was praised for his gallantry in the Seven Days Battles and the Battle of Antietam, but was demoted for insubordination after the disastrous defeat at the
Battle of Fredericksburg.

 As chief engineer of the Army of the Cumberland, he achieved recognition by restoring a supply line that saved that army from starvation and surrender, known
the “Cracker Line”, that helped Union troops to success in the Chattanooga Campaign in the autumn of 1863.

 Leading the first operation against Petersburg, Smith’s hesitation, possibly illness-related, cost the Union a prime opportunity for a quick end to the war,
and he was relieved of command.
</item 6>

<item 7>
Graph query row:
person: George Albert Smith Sr.
relationship: MEMBER_OF
connected_entity: The Church of Jesus Christ of Latter-day Saints
kind: church
role: eighth president
period: None
status: None
</item 7>

<item 8>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 8>

<item 9>
Albert Smith Medal
The Albert Smith Memorial Medal is the award given to the Man of the Match in the final of the Camanachd Cup, the blue riband trophy of the sport of shinty.

 It has been presented every year since 1972 by the Smith family of Fort William in honour of Albert Smith Sr. (Born in 1888, Lochuanagan, Fort Augustus.)

 The widow of his only son, Albert Smith Jr. currently presents the medal.

 Albert Sr.'s grandson, Victor Smith is a former player for Fort William Shinty Club but never won the medal despite featuring as a key player in several Fort
William wins.
</item 9>

<item 10>
Adam Fox
Adam Fox (1883 – 1977), Canon, was the Dean of Divinity at Magdalen College, Oxford.

 He was one of the first members of the "Inklings", a literary group which also included C.S. Lewis and J.R.R. Tolkien.

 Between 1938 and 1942 he was Professor of Poetry.

 Later he became Canon of Westminster Abbey and he is buried there in Poet's Corner.

 He was also Warden (Headmaster) of the famous Radley College.
</item 10>
=== END ITEM id=b116b150acd7a527 ===

=== ITEM id=7668c3c447aef553 ===
Gold sentences and facts:
Sentence 0: Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.
  Fact 0.0: Wangaratta Wahine is an album by the Australian band The Captain Matchbox Whoopee Band.
  Fact 0.1: Wangaratta Wahine was released in 1974.
  Fact 0.2: Wangaratta Wahine was the second album of The Captain Matchbox Whoopee Band.
Sentence 1:  He was declared an Australian Living Treasure by the National Trust of Australia in 1999.
  Fact 1.0: The subject of this sentence was declared an Australian Living Treasure by the National Trust of Australia.
  Fact 1.1: The subject of this sentence received this declaration in 1999.

Retrieved items:
<item 1>
The collection surveys music albums, artists, songs, and genres across rock, pop, metal, and electronic music. Highlights include Patty Griffin’s 2004 album
*Impossible Dream*; Edward Sharpe and the Magnetic Zeros’ 2013 album and 2010 single; the White Stripes’ rise, breakup, acclaimed *Elephant* (2003), and
Grammy-winning “Seven Nation Army”; and biographies of Meg White and Jack White. Other entries cover releases by Steve Winwood
</item 1>

<item 2>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 2>

<item 3>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 3>

<item 4>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 4>

<item 5>
Wangaratta Wahine
Wangaratta Wahine is an album by Australian band The Captain Matchbox Whoopee Band, released in 1974 and was their second album.

 It was recorded at top recording studio Armstrong Studios in Melbourne and the cover art was by famed Australian artist Michael Leunig.
</item 5>

<item 6>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 6>

<item 7>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
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
Live at Winterland 1978 (Avengers album)
Live at Winterland 1978 is a live album by the Avengers.

 It was released through online music stores on February 16, 2010.

 The album features a recording of their set on January 14, 1978 at the Winterland Ballroom in San Francisco, CA, when they opened for the Sex Pistols on what
would become the Sex Pistols final show, before reuniting years later.

 The Sex Pistols' set has been released on an album of the same name.

 Songs from this album were previously released on a couple 7" bootlegs titled "Penelope" and "Summer of Hate".
</item 9>

<item 10>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 10>
=== END ITEM id=7668c3c447aef553 ===

=== ITEM id=27766325509b7bdb ===
Gold sentences and facts:
Sentence 0:  It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of Superman called Nuclear Man.
  Fact 0.0: The film is the fourth and final film in the original Superman film series.
  Fact 0.1: The film is the first film in the original Superman series not produced by Alexander and Ilya Salkind.
  Fact 0.2: The film was produced by Golan-Globus' Cannon Films.
  Fact 0.3: The film was produced in association with Warner Bros.
  Fact 0.4: Gene Hackman returned as Lex Luthor in the film.
  Fact 0.5: Lex Luthor creates an evil solar-powered version of Superman called Nuclear Man.
Sentence 1: Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.
  Fact 1.0: Eugene Allen 'Gene' Hackman was born on January 30, 1930.
  Fact 1.1: Gene Hackman is a retired American actor.
  Fact 1.2: Gene Hackman is a retired American novelist.

Retrieved items:
<item 1>
Superman IV: The Quest for Peace
Superman IV: The Quest for Peace is a 1987 American superhero film directed by Sidney J. Furie, based on the DC Comics character Superman.

 It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but
rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of
Superman called Nuclear Man.
</item 1>

<item 2>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 2>

<item 3>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 3>

<item 4>
Alternative versions of Lex Luthor
The fictional character Lex Luthor has appeared in a number of media, always as the archenemy of Superman.

 Each version of the work typically establishes its own continuity, and sometimes introduces parallel universes, to the point where distinct differences in the
portrayal of the character can be identified.

 In addition, the DC Universe has been rewritten a number of times, establishing additional versions of the character.

 This article details and lists various versions of Lex Luthor depicted in works including DC Comics "Multiverse", "Elseworlds", television and film.
</item 4>

<item 5>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 5>

<item 6>
Lex Luthor: Man of Steel
Lex Luthor: Man of Steel (later collected as simply Luthor) is a five-issue monthly American comic book limited series written by Brian Azzarello and
illustrated by Lee Bermejo, which features Superman's nemesis Lex Luthor as the protagonist.
</item 6>

<item 7>
Charles A. Agemian
Charles A. Agemian (July 22, 1909 – March 30, 1996) was an American banker who took early retirement from Chase Manhattan Bank, where he was executive vice
president of operations, to become chairman and chief executive officer of the Hackensack Trust Co., which later was renamed Garden State National Bank.
</item 7>

<item 8>
Lionel Luthor
Lionel Luthor is a fictional character portrayed by John Glover in the television series "Smallville".

 The character was initially a recurring guest in season one, and became a series regular in season two and continued until being written out of the show in
season seven.

 The character returned to the show in season ten again as a recurring guest role as a parallel universe (Earth-2) version of the character.

 In "Smallville", Lionel Luthor is the father of Lex Luthor (Michael Rosenbaum), and founder and CEO of LuthorCorp.

 Lex Luthor's father was first introduced in "Superman" comics by Jerry Siegel in 1961 and has since appeared in other Superman-related media under different
names.

 "Smallville" is the first appearance in which the character has been an intricate part of a Superman adaptation.

 Series developers Alfred Gough and Miles Millar created Lionel Luthor for "Smallville" to provide an antithesis to the parenting style of Jonathan Kent (John
Schneider) and Martha Kent (Annette O'Toole).
</item 8>

<item 9>
Lex Luthor (Smallville)
Lex Luthor is a fictional character from the television series "Smallville".

 He was a series regular from the pilot episode until the season seven finale, and has been played continuously by Michael Rosenbaum, with various actors
portraying the character as a child throughout the series.

 The character of Lex Luthor, first created for comic books by Jerry Siegel and Joe Shuster in 1940 as nemesis of Superman, was adapted to television in 2001 by
Alfred Gough and Miles Millar—this is only the third time the character has been adapted to a live action television series.

 The character has also appeared in various literature based on the "Smallville" television series, none of which directly continues from or into the television
episodes.
</item 9>

<item 10>
Gene Hackman
Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.

 In a career spanning nearly five decades, Hackman was nominated for five Academy Awards, winning Best Actor in "The French Connection" and Best Supporting
Actor in "Unforgiven".

 He won three Golden Globes and two BAFTAs.
</item 10>
=== END ITEM id=27766325509b7bdb ===

=== ITEM id=26cd3470388568cb ===
Gold sentences and facts:
Sentence 0: Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.
  Fact 0.0: Ferdi Taygan was born on December 5, 1956
  Fact 0.1: Ferdi Taygan is a former professional tennis player
  Fact 0.2: Ferdi Taygan is from the United States
Sentence 1: Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.
  Fact 1.0: Mahesh Shrinivas Bhupathi was born on 7 June 1974
  Fact 1.1: Mahesh Shrinivas Bhupathi is a retired Indian professional tennis player

Retrieved items:
<item 1>
Ferdi Taygan
Ferdi Taygan (born December 5, 1956), is a former professional tennis player from the United States.

 He is of Turkish descent.
</item 1>

<item 2>
Graph query row:
person: Mahesh Bhupathi
birth_date: 7 June 1974
</item 2>

<item 3>
Mahesh Bhupathi
Mahesh Shrinivas Bhupathi (born 7 June 1974) is a retired Indian professional tennis player.

 In 1997, he became the first Indian to win a Grand Slam tournament (with Rika Hiraki).

 With his win at the Australian Open mixed doubles in 2006, he joined the elite group of eight tennis players who have achieved a career Grand Slam in mixed
doubles.

 He is also the founder of International Premier Tennis League.

 In December 2016, Bhupathi was appointed as India's next non-playing Davis Cup captain and took over the reins from Anand Amritraj in February 2017.
</item 3>

<item 4>
1982 Volvo Masters – Doubles
Peter Fleming and John McEnroe won in the final 6–4, 6–3 against Sherwood Stewart and Ferdi Taygan.
</item 4>

<item 5>
Mahesh Bhupathi Tennis Academy
Mahesh Bhupathi Tennis Academy is a tennis training facility started by Indian tennis player Mahesh Bhupathi.

 The Academy has facilities across 9 states in India as well as in the UAE.
</item 5>

<item 6>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 6>

<item 7>
Graph query row:
person: Ferdi Taygan
birth_date: December 5, 1956
</item 7>

<item 8>
1983 Donnay Indoor Championships – Doubles
Pavel Složil and Sherwood Stewart were the defending champions, but Složil did not participate this year.

 Stewart partnered Ferdi Taygan, losing in the first round.
</item 8>

<item 9>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 9>

<item 10>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 10>
=== END ITEM id=26cd3470388568cb ===

=== ITEM id=f24bce6b84e7b6cb ===
Gold sentences and facts:
Sentence 0:  Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.0: Reed is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.1: Mike Pringle is one of the players most often mentioned as being the greatest running back in CFL history.
  Fact 0.2: Johnny Bright is one of the players most often mentioned as being the greatest running back in CFL history.
Sentence 1: Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.
  Fact 1.0: Johnny D. Bright was born June 11, 1930.
  Fact 1.1: Johnny D. Bright died December 14, 1983.
  Fact 1.2: Johnny D. Bright was a professional Canadian football player.
  Fact 1.3: Johnny D. Bright played in the Canadian Football League.
Sentence 2:  He played college football at Drake University.
  Fact 2.0: Johnny D. Bright played college football at Drake University.

Retrieved items:
<item 1>
Richard Harris (American football)
Richard Drew Harris (January 21, 1948 – July 26, 2011) was an American football defensive end who played seven seasons in the National Football League.

 He was and All-American in 1970 for Grambling and was drafted in the first round (5th overall pick) of the 1971 NFL Draft by the Philadelphia Eagles.

 He was named All-Rookie in 1971, playing defensive end.

 Harris spent seven seasons as a defensive end in the NFL, the first three with the Philadelphia Eagles, the next two with the Chicago Bears and the final two
with the Seattle Seahawks.
</item 1>

<item 2>
Damon Allen
Damon Allen (born July 29, 1963) is a former professional quarterback who played in the Canadian Football League.

 He is currently second in all-time professional football passing yards and second in all-time CFL passing yards after he was surpassed for first place by the
Montréal Alouettes' Anthony Calvillo on October 10, 2011.

 He is pro-football's combined passing & rushing leader with a total of 84,301 yards.

 Allen retired as professional football's all-time leading passer with 72,381 passing yards after he surpassed Warren Moon's total of 70,553 yards (in both the
CFL and NFL combined) on September 4, 2006 in the annual Labour Day Classic.

 He also retired in third place in all-time CFL rushing yards with 11,920 yards, behind Mike Pringle and George Reed.

 The 2007 season marked Allen's twenty-third season in the CFL and he officially announced his retirement on May 28, 2008 at age 44.

 Allen is the younger brother of Pro Football Hall of Famer Marcus Allen.
</item 2>

<item 3>
George Reed (Canadian football)
George Robert Reed, CM, SOM (born October 2, 1939), is a former American college football and Canadian Football League player.

 Reed, along with Mike Pringle and Johnny Bright, is one of the players most often mentioned as being the greatest running back in CFL history.

 In November 2006, Reed was voted one of the CFL's Top 50 players (#2) of the league's modern era by Canadian sports network TSN.
</item 3>

<item 4>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 4>

<item 5>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 5>

<item 6>
Wilbert Montgomery
Wilbert Montgomery (born September 16, 1954) is a former American football player in the National Football League for nine years with the Philadelphia Eagles
and Detroit Lions.

 In the past, Montgomery has been the running backs and tight ends coach for St. Louis Rams (1997–2005), the running backs coach for the Detroit Lions
(2006-2007), the running backs coach for the Baltimore Ravens (2008–2013), and the running backs coach for the Cleveland Browns (2014–2015).
</item 6>

<item 7>
2,000-yard club
The 2,000-yard club is a group of seven National Football League (NFL) running backs that have rushed for 2,000 or more yards in a season.

 These seven rushing seasons rank as the highest single-season rushing totals in NFL history, and reaching the 2,000-yard mark is considered a significant
achievement for running backs.

 No running back has yet achieved this feat twice.

 The first 2,000-yard season was recorded in 1973 by Buffalo Bills running back O.J. Simpson.

 He is the only player to have surpassed 2,000 yards in a 14-game season, as all others occurred in 16-game seasons; he finished the season with 2,003 rushing
yards, averaging six yards per carry and an NFL-record 143.1 rushing yards per game.

 Los Angeles Rams running back Eric Dickerson, who had broken the single-season rookie rushing record in 1983, recorded the second 2,000-yard season in 1984.

 Dickerson rushed for 2,105 yards, the current NFL rushing record, and averaged 131.6 rushing yards per game.
</item 7>

<item 8>
Marc Mueller
Marc Mueller (born April 15, 1989) is the current running backs coach of the Calgary Stampeders of the Canadian Football League (CFL).

 In college, he played quarterback for the Regina Rams of the CIS.

 He signed as an undrafted free agent with the Edmonton Eskimos of the CFL on May 9, 2011 following the 2011 CFL Draft.

 He was released on June 20, 2011 in order to play his final year for the Rams.

 Mueller is the grandson of former CFL quarterback and coach Ron Lancaster.
</item 8>

<item 9>
Skip Peete
Skip Peete (born January 30, 1963) is an American football coach who serves as the current running backs coach for the Los Angeles Rams of the National Football
League (NFL).

 Prior to the Rams, he coached running backs for the Chicago Bears, Dallas Cowboys, and Oakland Raiders.

 He also has experience in college, coaching running backs for two years at UCLA, wide receivers at Michigan State and Rutgers, and both running backs and wide
receivers at the University of Pittsburgh.
</item 9>

<item 10>
Dolph Pulliam
Adolphus "Dolph" Pulliam (born March 14, 1946 in West Point, Mississippi) is a former American basketball player.

 He played collegiately at Drake University.

 He was named to Drake University's All-Century basketball team.

 He helped lead the Drake men's basketball team to the 1969 Final Four and a third-place finish.

 On 7 February 2009, Drake University retired Pulliam's #5 jersey that he wore for the Bulldogs.
</item 10>
=== END ITEM id=f24bce6b84e7b6cb ===

=== ITEM id=e0662c121cbcef8d ===
Gold sentences and facts:
Sentence 0: Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.
  Fact 0.0: Edward Benjamin Britten, Baron Britten, lived from 22 November 1913 to 4 December 1976.
  Fact 0.1: Edward Benjamin Britten, Baron Britten, was an English composer.
  Fact 0.2: Edward Benjamin Britten, Baron Britten, was an English conductor.
  Fact 0.3: Edward Benjamin Britten, Baron Britten, was an English pianist.
Sentence 1: Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.
  Fact 1.0: Béla Viktor János Bartók lived from 25 March 1881 to 26 September 1945.
  Fact 1.1: Béla Viktor János Bartók was a Hungarian composer.
  Fact 1.2: Béla Viktor János Bartók was a Hungarian pianist.
  Fact 1.3: Béla Viktor János Bartók was a Hungarian ethnomusicologist.

Retrieved items:
<item 1>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 1>

<item 2>
Béla Bartók Music High School
Béla Bartók Music High School (Bartók Béla Zeneművészeti Szakközépiskola) is situated in the Palace of Music (Zenepalota) in Bartók square Miskolc, Hungary.

It is a famous Music school named after the famous Hungarian composer Béla Bartók.

It was founded in 1966.
</item 2>

<item 3>
List of string quartets by Béla Bartók
1 and 28 both owe a great deal to Bartók's quartets (; ), György Ligeti, whose two string quartets both owe a great deal to Bartók’s quartets (; ), Bruno
Maderna , George Perle, who credits the Bartók Fourth and Fifth Quartets as precedents for his use of arrays of chords related to one another by different types
of symmetry , Walter Piston (; ), Kim Dzmitrïyevich Tsesakow , Wilfried Westerlinck , Stefan Wolpe, who explained in a public lecture how he had derived ideas
from Bartók’s Fourth Quartet , and Xu Yongsan .
</item 3>

<item 4>
Benjamin Britten
Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.

 He was a central figure of 20th-century British classical music, with a range of works including opera, other vocal music, orchestral and chamber pieces.

 His best-known works include the opera "Peter Grimes" (1945), the "War Requiem" (1962) and the orchestral showpiece "The Young Person's Guide to the Orchestra"
(1945).
</item 4>

<item 5>
List of string quartets by Béla Bartók
The Hungarian composer Béla Bartók wrote six string quartets, each for the usual forces of two violins, viola and cello.

 Notable composers who have been influenced by them include Benjamin Britten, particularly in the Sonata in C for Cello and Piano (; ), Elliott Carter, who
refers in the opening of his own First String Quartet to Bartók’s Sixth Quartet , Chen Yi , Edison Denisov, whose Second Quartet is closely related to Bartók’s
Fifth Quartet , Franco Donatoni, who was deeply impressed when he heard a broadcast of Bartók's Fourth Quartet , Robert Fripp, who mentions them as an influence
upon King Crimson , Miloslav Ištvan , György Kurtág, whose Opp.
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
String Quartet No. 6 (Bartók)
The String Quartet No. 6, Sz.

 114, BB 119, was the final string quartet that Béla Bartók wrote before his death.

 It was begun in August 1939 in Saanen, Switzerland, where Bartók was a guest of his patron, the conductor Paul Sacher.

 Shortly after he completed the Divertimento for String Orchestra on the 17th, he started on a commission for his friend, the violinist Zoltán Székely.

 Székely was acting as intermediary for the "New Hungarian Quartet", who had given the Budapest premiere of the String Quartet No. 5.

 With the outbreak of World War II and his mother's illness, Bartók returned to Budapest, where the quartet was finished in November.

 After his mother's death, Bartók decided to leave with his family for the United States.
</item 7>

<item 8>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 8>

<item 9>
List of compositions by Béla Bartók
This aspires to be a complete list of compositions by Béla Bartók.

 The catalogue numbering by András Szőllősy (Sz.)

, László Somfai (BB.)

 and Denijs Dille (DD.)

 are provided, as well as Bartók's own opus numbers.

 Note that Bartók started three times anew with opus numbers, here indicated with "(list 1)", "(list 2)" and "(list 3)" respectively.

 The pieces from the third listing are by far best known; opus lists 1 and 2 are early works.

 The year of composition and instrumentation (including voice) are included.

 See the main article on Béla Bartók for more details.
</item 9>

<item 10>
Suite, Op. 14 (Bartók)
The Suite, Op. 14, Sz.

 62, BB 70 is a piece for solo piano written by Béla Bartók.

 It was written in February 1916, published in 1918, and debuted by the composer on April 21, 1919, in Budapest.

 The Suite is one of Bartók's most significant works for piano, only comparable with his 1926 Piano Sonata.

 Though much of Bartók's work makes frequent use of Eastern European folk music, this suite is one of the few pieces without melodies of folk origin.

 However, Romanian, Arabic, and North African rhythmic influences can still be found in some movements.

 Originally intending the suite to be a five-movement work, Bartók later decided against the idea and discarded the second movement, the "Andante", which was
published only posthumously in the October 1955 issue of "Új Zenei Szemle" (New Musical Review).
</item 10>
=== END ITEM id=e0662c121cbcef8d ===
