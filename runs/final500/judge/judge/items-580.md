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

=== ITEM id=43bf9a6ab06fd870 ===
Gold sentences and facts:
Sentence 0: Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.
  Fact 0.0: Andrew and Jeremy Get Married is a 2004 film.
  Fact 0.1: Andrew and Jeremy Get Married is a British documentary film.
  Fact 0.2: Andrew and Jeremy Get Married was written by Don Boyd.
  Fact 0.3: Andrew and Jeremy Get Married was directed by Don Boyd.
  Fact 0.4: Andrew and Jeremy Get Married was made for the BBC.
Sentence 1: The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.
  Fact 1.0: The Last Lions is a 2011 film.
  Fact 1.1: The Last Lions is an African nature documentary film.
  Fact 1.2: The Last Lions is by the National Geographic Society.
  Fact 1.3: The Last Lions was videotaped by Dereck and Beverly Joubert.
  Fact 1.4: The Last Lions was directed by Dereck and Beverly Joubert.

Retrieved items:
<item 1>
The Bears and I
The Bears and I is a 1974 American drama film directed by Bernard McEveety and written by John Whedon.

 The film stars Patrick Wayne, Chief Dan George, Andrew Duggan, Michael Ansara and Robert Pine.

 The film was released on July 31, 1974, by Buena Vista Distribution.
</item 1>

<item 2>
Tyler Perry's For Better or Worse
Tyler Perry's For Better or Worse is an American television sitcom created, written and executive produced by Tyler Perry.

 The series is based on and is a TV spin-off to Perry's 2007 film "Why Did I Get Married?

" and its 2010 sequel "Why Did I Get Married Too?

".

 The show premiered on TBS on November 25, 2011 and ended on July 22, 2017.

 Led by the comical, over-the-top antics of Marcus and Angela Williams, the ensemble follows three couples: Marcus and Angela, Joseph and Leslie, and Richard
and Keisha who are at various stages of their relationships as they go through the ups-and-downs of married life and dating.

 The third season of the series premiered on September 18, 2013, on the Oprah Winfrey Network.
</item 2>

<item 3>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 3>

<item 4>
Andrew and Jeremy Get Married
Andrew and Jeremy Get Married is a 2004 British documentary film written and directed by Don Boyd for the BBC.

 It tells the story of two Englishmen, Andrew Thomas and Jeremy Trafford, as they plan for their commitment ceremony.

 Originally commissioned for the "BBC Storyville" series, the film premiered at the 2004 Toronto International Film Festival.
</item 4>

<item 5>
Get Married 3
Get Married 3 is an Indonesian romantic comedy directed by Monty Tiwa and released in 2011.

 A sequel to Hanung Bramantyo's "Get Married" and "Get Married 2", it stars Nirina Zubir and Fedi Nuril as a married couple attempting to raise their triplets
while under intense pressure from their family and friends.

 The film was a commercial success and received favourable reviews in "The Jakarta Post" and "Suara Karya".
</item 5>

<item 6>
The Compulsory Husband
The Compulsory Husband is a 1930 British comedy film directed by Monty Banks and Harry Lachman and starring Banks, Lillian Manton and Clifford Heatherley.

 It was based on a novel by John Glyder.
</item 6>

<item 7>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 7>

<item 8>
The Lion (2010 novel)
The Lion is a 2010 novel by American author Nelson DeMille.

 It is the fifth of DeMille's novels to feature Detective John Corey, now working as a contractor for the fictional FBI Anti-Terrorist Task Force in New York
City.

 The novel is the sequel to "Wild Fire".

 The Lion is followed by DeMille’s 2012 novel, "The Panther".
</item 8>

<item 9>
Two Brides and a Baby
Two Brides and a Baby is a 2011 Nigerian romantic drama film directed by Teco Benson, starring Keira Hewatch, Kalu Ikeagwu, OC Ukeje, Chelsea Eze, Stella
Damasus-Aboderin and Okey Uzoeshi.

 It premiered on November 17, 2011.

 It received awards and nominations at Africa Movie Academy Awards, Best of Nollywood Awards and Africa Magic Viewers Choice Awards.
</item 9>

<item 10>
Richard T. Jones
Richard Timothy Jones (born January 16, 1972), known professionally as Richard T. Jones, is an American actor.

 Jones is best known for his portrayals of Laveinio in the dramatic film "The Wood" and Mike of the dramatic films "Why Did I Get Married?

" and "Why Did I Get Married Too?

".
</item 10>
=== END ITEM id=43bf9a6ab06fd870 ===

=== ITEM id=42a0e17fafa737e8 ===
Gold sentences and facts:
Sentence 0:  It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
  Fact 0.0: The described act consists of a duet.
  Fact 0.1: Dave Hemingway is a vocalist featured in the duet.
  Fact 0.2: Briana Corrigan is a vocalist featured in the duet.
Sentence 1: Briana Corrigan (born 30 May 1965) is a Northern Irish singer.
  Fact 1.0: Briana Corrigan was born on 30 May 1965.
  Fact 1.1: Briana Corrigan is a Northern Irish singer.

Retrieved items:
<item 1>
Takes a Little Time (Total Contrast song)
"Takes a Little Time" is a 1985 dance hit by British, techno-soul, duo, Total Contrast.

 Their debut American release went to number one on the U.S. dance chart for one week.

 "Takes a Little Time" did not make the Hot 100 but, peaked at number eighty on the R&B singles chart.
</item 1>

<item 2>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 2>

<item 3>
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
</item 3>

<item 4>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 4>

<item 5>
The Time Is Now (John Cena song)
"The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.

 It is the current entrance theme Cena uses as he makes his entrance to the ring.

 Currently wrestling for WWE, Cena recorded the song in 2005 for his debut studio album, "You Can't See Me".
</item 5>

<item 6>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 6>

<item 7>
Breathing Your Love
"Breathing Your Love" is a song by the Swedish singer Darin featuring vocals by singer Kat DeLuna and the first single from "Flashback".

 The song co-written by Darin with RedOne, Bilal Hajji and Novel was released to radio stations and as a digital download in Sweden on 8 October 2008.

 The single was released in Finland and is also the first single from Darin to be released in the United Kingdom in January 2010.

 In October 2009 Swedish astronaut Christer Fuglesang took 2 copies of the single into space with him, as his daughter is a Darin fan, a photo of the disc can
be seen with earth in the distance can be found on the official Darin website.
</item 7>

<item 8>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 8>

<item 9>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 9>

<item 10>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 10>
=== END ITEM id=42a0e17fafa737e8 ===

=== ITEM id=70b365374c9c7dac ===
Gold sentences and facts:
Sentence 0: The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic Beach.
  Fact 0.0: The Nassau Herald is a weekly newspaper.
  Fact 0.1: The Nassau Herald serves the Five Towns communities of Nassau County.
  Fact 0.2: The Five Towns communities of Nassau County are Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood, and Atlantic Beach.
Sentence 1: Lawrence is a village in Nassau County, New York in the United States.
  Fact 1.0: Lawrence is a village.
  Fact 1.1: Lawrence is in Nassau County, New York, United States.
Sentence 2:  As of the 2010 United States Census, the village population was 6,483.
  Fact 2.0: As of the 2010 United States Census, the population of the village of Lawrence was 6,483.

Retrieved items:
<item 1>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 1>

<item 2>
The Hudson Reporter
The Hudson Reporter is a newspaper chain based in Hudson County, New Jersey.

 It is the only weekly newspaper chain in Hudson County and one of only two newspaper companies in this busy metropolitan area.

 "The Hudson Reporter" publications focus on local politics and community news.

 In addition to articles written by the staff, the papers print readers' letters to the editor.

 The oldest newspaper in the chain is the "Hoboken Reporter", founded in 1983.
</item 2>

<item 3>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 3>

<item 4>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 4>

<item 5>
Anton Media Group
Anton Media Group, formerly Anton Community Newspapers, and also known by its legal name Long Island Community Newspapers, Inc. is a print media company based
in Mineola, NY on Long Island in Nassau County and produces 17 weekly newspapers in Nassau County as well as "Long Island Weekly" and special sections such as
Healthy Living, Camp & School, and Dining Guides.

 Anton Community Newspapers is one of the largest privately owned newspaper companies in New York State.
</item 5>

<item 6>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 6>

<item 7>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 7>

<item 8>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 8>

<item 9>
Floral Park, New York
Floral Park is an incorporated village in Nassau County, New York, United States, on Long Island.

 The neighborhood of Floral Park in the New York City borough of Queens, is adjacent to the village.

 The village is at the western border of Nassau County, and is located mainly in the Town of Hempstead, while the section north of Jericho Turnpike is within
the Town of North Hempstead.

 The population as of the US Census of 2010 is 15,863.
</item 9>

<item 10>
Nassau Herald
The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic
Beach.

 It is part of the Long Island Herald newspaper chain, which includes The Jewish Star and The Oyster Bay Guardian is owned by Richner Communications, and covers
Nassau County, New York.
</item 10>
=== END ITEM id=70b365374c9c7dac ===

=== ITEM id=57949396a8b4c009 ===
Gold sentences and facts:
Sentence 0:  It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
  Fact 0.0: This song is most known through the 1989 single by Kathy Mattea.
  Fact 0.1: The 1989 single by Kathy Mattea was released in conjunction with her album Willow in the Wind.
  Fact 0.2: This song was first recorded and released on the 1987 Don Williams album Traces.
  Fact 0.3: This song was also released in 1988 by Clark's husband on his album Old Friends.
Sentence 1: This is a detailed discography for American country music artist Don Williams.
  Fact 1.0: The discography is detailed.
  Fact 1.1: The discography is for American country music artist Don Williams.

Retrieved items:
<item 1>
Every Beat of My Heart (song)
"Every Beat of My Heart" is a rhythm and blues song by Johnny Otis.

 It was first recorded in 1954 by his group, The Royals (later to be known as The Midnighters).

 In 1961 Gladys Knight & the Pips recorded the song for their debut single on the Vee-Jay label.

 Credited to The Pips, it was the first of eleven releases by the group to make it to number one on the R&B/soul chart.

 It was also the group's first top ten on the "Billboard" Hot 100.

 "Every Beat of My Heart" was first recorded for the Huntom label, who later sold the master to Vee-Jay.

 At the time of the song's release, The Pips were on the Fury label where they re-recorded the song without piano.

 In an unusual occurrence, the Fury recording of the song also made the top twenty on the R&B Sides chart and also made the Hot 100.
</item 1>

<item 2>
Come On (Earl King song)
"Come On" (often called "Let the Good Times Roll") is a song written by New Orleans rhythm and blues artist Earl King.

 He first recorded the song as "Darling Honey Angel Child" in 1960 for the Ace Records subsidiary Rex.

 Later that year, he recorded it as a two-part song for Imperial Records using some new lyrics.

 Retitled "Come On", it was released in 1960 with "Come On - Part I” as the A-side backed with “Come On - Part II” (Imperial 5713).
</item 2>

<item 3>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 3>

<item 4>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 4>

<item 5>
Can We Talk
"Can We Talk" is a song recorded by American R&B singer Tevin Campbell and composed and produced by Babyface.

 It was the first single to be released from his double platinum second release "I'm Ready".

 The song hit top ten on the pop charts peaking at number nine on the "Billboard" Hot 100 and spent a total of three weeks at number one on the US R&B chart.

 It sold 500,000 copies and earned a gold certification from the Recording Industry Association of America.

 The song was also Nominated for a Grammy Award for Best Rhythm & Blues Vocal Performance - Male.

 It was also nominated and later won the Soul Train Music Award for Best R&B/Soul Single - Male ("Can We Talk").

 Cover versions of the hit song have been done by The Whispers, Sanchez, and Kirk Whalum (on his 2005 album "Kirk Whalum Performs the Babyface Songbook").

 The song was later covered by British boyband Code Red in 1996, for their debut album "Scarlet" and was released as the album's lead single becoming a modest
hit and reaching #1 in Asia.
</item 5>

<item 6>
Hate to See Your Heart Break
"Hate to See Your Heart Break" is a song by American rock band Paramore, recorded for their 2013 self-titled fourth album "Paramore".

 It was re-recorded to feature vocals by Joy Williams (formerly of The Civil Wars) for the 2014 deluxe edition of the album, the first time Paramore has
collaborated with another artist or group on a studio recording.
</item 6>

<item 7>
Refugees of the Heart
Refugees of the Heart is the sixth solo studio album by Steve Winwood, released in 1990.

 The album contained the hit single, "One and Only Man", which topped the Mainstream Rock Tracks chart and saw the return of former Traffic bandmate Jim Capaldi
to Winwood's songwriting team.

 By coincidence, a Traffic reunion followed in 1994.

 Because of that collaboration, Winwood would not record another solo album until late 1997.

 Also "I Will Be Here" and "Another Deal Goes Down" were released as singles.

 Winwood stated about the closing track, “In The Light Of Day”: 'When Will and I wrote the song (.

.)

 it was our idea of what Nelson Mandela's dream was, while he was in prison.

 It was really just a fantasy of ours, but that’s what we based the song on.'
</item 7>

<item 8>
Hjärtats trakt
Hjärtats trakt was released on 20 April 1993 and is a compilation album from Swedish pop artist Per Gessle.

 The album includes songs taken from Gessle's solo albums "Per Gessle" and "Scener".

 It was also released under license by the British record company Pickwick and it is the only Gessle album released by another label than EMI.

 In 1998 the album was re-released by the Dutch record company Disky Communications B.V. which is a subsidiary of EMI.
</item 8>

<item 9>
Night Is Fallin' in My Heart
"Night Is Fallin' in My Heart" is a song written by Dennis Linde.

 The song was first recorded by Country music artist J.P. Pennington for his 1991 album, "Whatever It Takes".

 It was later recorded and released as a single in October 1994 by American country music group Diamond Rio.

 It was the second single released from their third album, "Love a Little Stronger".

 It peaked at number 9 in the United States, and number 6 in Canada.
</item 9>

<item 10>
Can't Help Myself
"Can't Help Myself" is the first single released by the Australian synthpop/rock band Flowers, later known as Icehouse.

 It was released in May 1980 as a 7" vinyl single on independent label, Regular Records, five months ahead of debut album "Icehouse".

 A 10" vinyl single was released in July and had a cover depicting individual images of band members diagonally across the band's name and the single's title
(see infobox at right middle).

 It peaked at #10 on the Australian Singles Charts.
</item 10>
=== END ITEM id=57949396a8b4c009 ===

=== ITEM id=734b54915bd17eed ===
Gold sentences and facts:
Sentence 0: A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.
  Fact 0.0: A Summer in the Cage is a documentary film.
  Fact 0.1: A Summer in the Cage was released in 2007.
  Fact 0.2: A Summer in the Cage is about a man's experiences with bipolar disorder.
Sentence 1: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.
  Fact 1.0: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary.
  Fact 1.1: American Hardcore: The History of American Punk Rock 1980-1986 was directed by Paul Rachman.
  Fact 1.2: American Hardcore: The History of American Punk Rock 1980-1986 was written by Steven Blush.

Retrieved items:
<item 1>
I Against I
I Against I is the third studio album by the American hardcore punk band Bad Brains.

 It was released in November 1986 through SST Records with the catalog number SST 065.

 The best-selling album in the band's catalog, "I Against I" is a critically acclaimed album that mixes American hardcore punk with funk, soul, reggae and heavy
metal.

 It is also included in the book "1001 Albums You Must Hear Before You Die".

 The album featured an MTV video for the title track directed by Paul Rachman who later went on to direct the documentary feature film "American Hardcore".
</item 1>

<item 2>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 2>

<item 3>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 3>

<item 4>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 4>

<item 5>
Actuality film
The actuality film is a non-fiction film genre that, like the documentary film, uses footage of real events, places, and things, yet unlike the documentary is
not structured into a larger argument, picture of the phenomenon or coherent whole.

 In practice, actuality films preceded the emergence of the documentary.

 During the era of early cinema, actualities—usually lasting no more than a minute or two and usually assembled together into a program by an exhibitor—were
just as popular and prominent as their fictional counterparts.

 The line between "fact" and "fiction" was not so sharply drawn in early cinema as it would become after the documentary came to serve as the predominant
non-fiction filmmaking form.

 An actuality film is not like a newspaper article so much as it is like the still photograph that is published along with the article, with the major
difference being that it moves.

 Apart from the traveling actuality genre, actuality is one film genre that remains strongly related to still photography.
</item 5>

<item 6>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 6>

<item 7>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 7>

<item 8>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 8>

<item 9>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 9>

<item 10>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 10>
=== END ITEM id=734b54915bd17eed ===

=== ITEM id=b41e495b5a9c3900 ===
Gold sentences and facts:
Sentence 0:  He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.
  Fact 0.0: He played college football at Hampton University.
  Fact 0.1: He attended Riverdale High School.
  Fact 0.2: Riverdale High School is in Riverdale, Georgia.
Sentence 1: Riverdale is a city in Clayton County, Georgia, United States.
  Fact 1.0: Riverdale is a city.
  Fact 1.1: Riverdale is in Clayton County, Georgia.
  Fact 1.2: Clayton County, Georgia is in the United States.

Retrieved items:
<item 1>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 1>

<item 2>
Two Rivers High School (Wisconsin)
Two Rivers High School is a high school located in Two Rivers, Wisconsin.

 It was built in 2001 to replace the former Washington High School in downtown Two Rivers.

 The School is located at 4519 Lincoln Avenue Wisconsin Highway 42 on the outskirts of Two Rivers.

 The school athletically takes part in the Eastern Wisconsin Conference.

 Conference rivals include Plymouth High School and Kiel High School.

 A total of 24 Credits and 20 Community Service Hours are required in order to graduate from the school.
</item 2>

<item 3>
Riverdale, Georgia
Riverdale is a city in Clayton County, Georgia, United States.

 The population was 15,134 at the 2010 census, up from 12,478 in 2000.

 Riverdale is a suburb just south of Hartsfield–Jackson Atlanta International Airport and is part of the Atlanta metropolitan area.
</item 3>

<item 4>
College of Coastal Georgia
The College of Coastal Georgia (also known as CCGA) is a public college located at historical Brunswick, Georgia, United States and the surrounding historical
Golden Isles.

 The College was established in 1961 and opened in 1964, making it one of Georgia's newest state colleges.

 The college transitioned from a two-year community college into a four-year comprehensive institution and conferred its first baccalaureate degrees on May 7,
2011.
</item 4>

<item 5>
Two Rivers High School (Arkansas)
Two Rivers High School is a public high school for students in grades 7 through 12 located in Ola, Arkansas, United States.

 Two Rivers High School is administered by the Two Rivers School District.

 The Two Rivers High School and school district serves 650 sqmi of rural communities including Ola, Plainview, Rover, and Casa.

 Two Rivers High School is one of four high schools within Yell County and the sole high school administered by the Two Rivers School District.
</item 5>

<item 6>
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
</item 6>

<item 7>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 7>

<item 8>
Michael Ola
Michael Ola (born April 19, 1988) is an American football offensive tackle for the Buffalo Bills of the National Football League (NFL).

 He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.

 He has also been a member of the Jacksonville Sharks, Montreal Alouettes, Miami Dolphins, Chicago Bears, San Diego Chargers, Detroit Lions, Seattle Seahawks
and New York Giants.
</item 8>

<item 9>
Brunswick Old Town Historic District
Brunswick Old Town Historic District is a historic district in Brunswick, Georgia.

 It was added to the National Register of Historic Places on April 26, 1979 and includes an area bounded by 1st Street, Bay Street, New Bay Street, H Street,
and Cochran Street (4th Ave. and G St., according to one source).

 Brunswick is one of Georgia's two deep-water ports and is the mainland city associated with the Golden Isles of Georgia, at the junction of I-95 and US 82.
</item 9>

<item 10>
Rich Central High School
Rich Central High School or RCHS is a public four-year high school located in Olympia Fields, Illinois, a southern suburb of Chicago, Illinois, in the United
States.

 The Rich Central Campus serves the cities of Matteson, Richton Park, Country Club Hills, Chicago Heights, Tinley Park, and parts of Olympia Fields.

 It is a part of Rich Township District 227, which also includes Rich East High School and Rich South High School.

 Although the school is located in Olympia Fields, it does not serve the entire village.

 Some Olympia Fields students attend high school at Homewood-Flossmoor High School and Bloom High School, depending on the subdivision where the student
resides.
</item 10>
=== END ITEM id=b41e495b5a9c3900 ===

=== ITEM id=02fa4d8341fb8c2d ===
Gold sentences and facts:
Sentence 0: The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.
  Fact 0.0: The Appenzeller Sennenhund is a medium-size breed of dog.
  Fact 0.1: The Appenzeller Sennenhund is one of the four regional breeds of Sennenhund-type dogs.
  Fact 0.2: The Sennenhund-type dogs come from the Swiss Alps.
Sentence 1:  The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
  Fact 1.0: The name Sennenhund refers to people called "Senn".
  Fact 1.1: The Senn people are herders in the Appenzell region of Switzerland.
Sentence 2: The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.
  Fact 2.0: The Drentsche Patrijshond is a versatile spaniel-type hunting dog.
  Fact 2.1: The Drentsche Patrijshond is from the Dutch province of Drenthe.

Retrieved items:
<item 1>
Bernese Mountain Dog
The Bernese Mountain Dog (German: "Berner Sennenhund" ) is a large-sized breed of dog, one of the four breeds of Sennenhund-type dogs from the Swiss Alps.

 The name "Sennenhund" is derived from the German "Senne" ("alpine pasture") and "Hund" ("dog"), as they accompanied the alpine herders and dairymen called
"Senn".

 "Berner" (or "Bernese" in English) refers to the area of the breed’s origin, in the canton of Bern.

 This mountain dog was originally kept as a general farm dog.

 Large Sennenhunde in the past were also used as draft animals, pulling carts.

 The breed was officially established in 1907.

 In 1937, the American Kennel Club recognized it; today, the club classifies it as a member of the Working Group.
</item 1>

<item 2>
Schutzhund
Schutzhund (German for "protection dog") is a dog sport that was developed in Germany in the early 1900s as a breed suitability test for the German Shepherd
breed.

 The test would determine if the dog displayed the appropriate traits and characteristics of a proper working German Shepherd.

 Today, it is used as a sport where many breeds other than German Shepherd Dogs can compete, but it is such a demanding test that few dogs can pass.
</item 2>

<item 3>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 3>

<item 4>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 4>

<item 5>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 5>

<item 6>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 6>

<item 7>
Entlebucher Mountain Dog
The Entlebucher Sennenhund or Entlebucher Mountain Dog is a medium-sized herding dog, it is the smallest of the four Sennenhunds, a dog type that includes four
regional breeds.

 The name Sennenhund refers to people called "Senn", herders in the Swiss Alps.

 Entlebuch is a region in the canton of Lucerne in Switzerland.

 The breed is also known in English as the Entelbuch Mountain Dog, Entelbucher Cattle Dog, and similar combinations.
</item 7>

<item 8>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 8>

<item 9>
Appenzeller (chicken)
The Appenzeller is a breed of chicken originating in Appenzell region of Switzerland.

 The Appenzeller comes in two varieties.

 The Spitzhauben variety, meaning "pointed hood", has a V-comb and feather crests in males and females.

 The word 'spitzhauben' derives from the frilly hat worn by the women in the Appenzeller region in Switzerland.

 The breed was imported into America by a doctor.

 The Barthuhner ("bearded hen") has a rose comb and no crest.

 Both types appear in either black, golden spangled and silver spangled plumage.

 They are mostly a show breed, but are decent egg layers.

 They lay small white eggs about 5 times per week.
</item 9>

<item 10>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 10>
=== END ITEM id=02fa4d8341fb8c2d ===

=== ITEM id=f1639adfb13b65e7 ===
Gold sentences and facts:
Sentence 0: The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.
  Fact 0.0: The Long–Allen Bridge is a truss bridge.
  Fact 0.1: The Long–Allen Bridge is in Shreveport, Louisiana.
  Fact 0.2: The Long–Allen Bridge is named for Louisiana governor Huey Long.
  Fact 0.3: The Long–Allen Bridge is named for Louisiana governor Oscar K. Allen.
Sentence 1:  Opened in 1933, it carries U.S. Route 80 across the Red River.
  Fact 1.0: The Long–Allen Bridge opened in 1933.
  Fact 1.1: The Long–Allen Bridge carries U.S. Route 80 across the Red River.
Sentence 2: U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.
  Fact 2.0: U.S. Route 80 (US 80) is an east–west United States Numbered Highway.
  Fact 2.1: Much of U.S. Route 80 was once part of the early auto trail known as the Dixie Overland Highway.
Sentence 3:  The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
  Fact 3.0: The eastern terminus of U.S. Route 80 is in Tybee Island, Georgia.
  Fact 3.1: The eastern terminus of U.S. Route 80 is at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street.
  Fact 3.2: The eastern terminus of U.S. Route 80 is near the Atlantic Ocean.

Retrieved items:
<item 1>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 1>

<item 2>
Long–Allen Bridge (Shreveport)
The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.

 Opened in 1933, it carries U.S. Route 80 across the Red River.

 It is also known as the Texas Street Bridge.
</item 2>

<item 3>
U.S. Route 80
U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.

 As the "0" in the route number indicates, it was originally a cross-country route, from the Pacific Ocean to the Atlantic Ocean.

 However, the entire segment west of Dallas, Texas, has been decommissioned in favor of various Interstate Highways and state highways.

 Currently, the highway's western terminus is at an interchange with Interstate 30 (I-30) on the Dallas–Mesquite, Texas city line.

 The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
</item 3>

<item 4>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 4>

<item 5>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 5>

<item 6>
Rocky Creek Bridge No. 01089
Rocky Creek Bridge No. 01089, also known as Ben Jones Bridge, is a concrete highway bridge spanning Rocky Creek along the Pacific Ocean coast of the U.S. state
of Oregon.

 The bridge crosses a gorge near the creek's mouth, about 2 mi south of Depoe Bay in Lincoln County.

 Built in 1927, the bridge originally carried U.S. Route 101 (the Oregon Coast Highway) over the creek.

 After a stretch of the highway was relocated to make it straighter, the abandoned piece of the old highway became Otter Crest Loop Road, a bypass west of the
new highway.
</item 6>

<item 7>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 7>

<item 8>
Low Bridge (song)
The popular song "Low Bridge, Everybody Down" was written in 1905 by Thomas S. Allen after Erie Canal barge traffic was converted from mule power to engine
power, raising the speed of traffic.

 Also known as "Fifteen Years on the Erie Canal", "Fifteen Miles on the Erie Canal", "Erie Canal Song", and "Mule Named Sal", the song memorializes the years
from 1825 to 1880 when the mule barges made boomtowns out of Utica, Rome, Syracuse, Rochester, and Buffalo, and transformed New York into the Empire State.

 The tune is sadly nostalgic.
</item 8>

<item 9>
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 9>

<item 10>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 10>
=== END ITEM id=f1639adfb13b65e7 ===

=== ITEM id=9f4d1fb667b23b6e ===
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
After Laughter
After Laughter is the fifth studio album by American rock band Paramore.

 It was released on May 12, 2017, through Fueled by Ramen as a follow-up to "Paramore", their 2013 self-titled album.

 The album was produced by guitarist Taylor York alongside previous collaborator, Justin Meldal-Johnsen.

 It is the band's first album since the return of drummer Zac Farro, who left the band with his brother Josh in 2010, and the departure of former bassist Jeremy
Davis, who left the band in 2015.

 "After Laughter" represents a complete departure from the usual pop punk and alternative rock sound of their previous releases.

 The album touches on themes of exhaustion, depression and anxiety, contrasting the upbeat and vibrant sound of the record.
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
=== END ITEM id=9f4d1fb667b23b6e ===

=== ITEM id=b6beaca87ec369a2 ===
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
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 1>

<item 2>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 2>

<item 3>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 3>

<item 4>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 4>

<item 5>
Sonya Scarlet
Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
</item 5>

<item 6>
Terri Nunn
Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.

 She is best known as the lead vocalist of the new wave/synthpop band Berlin.
</item 6>

<item 7>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 7>

<item 8>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 8>

<item 9>
Sonya Yoncheva
Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
</item 9>

<item 10>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 10>
=== END ITEM id=b6beaca87ec369a2 ===

=== ITEM id=bd74ffc993c05278 ===
Gold sentences and facts:
Sentence 0: The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between 1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
  Fact 0.0: The New York – New Jersey Line War is also known as the N.J. Line War.
  Fact 0.1: The New York – New Jersey Line War refers to a series of skirmishes and raids.
  Fact 0.2: The skirmishes and raids took place for over half a century.
  Fact 0.3: The skirmishes and raids took place between 1701 and 1765.
  Fact 0.4: The skirmishes and raids took place at the disputed border between two American colonies.
  Fact 0.5: The two American colonies were the Province of New York and the Province of New Jersey.
Sentence 1: The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.
  Fact 1.0: The Province of New York existed from 1664 to 1776.
  Fact 1.1: The Province of New York was a British proprietary colony.
  Fact 1.2: The Province of New York later became a royal colony.
  Fact 1.3: The Province of New York was on the northeast coast of North America.

Retrieved items:
<item 1>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 1>

<item 2>
New York – New Jersey Line War
The New York – New Jersey Line War (also known as the N.J. Line War) refers to a series of skirmishes and raids that took place for over half a century between
1701 and 1765 at the disputed border between two American colonies, the Province of New York and the Province of New Jersey.
</item 2>

<item 3>
Province of New York
The Province of New York (1664–1776) was a British proprietary colony and later royal colony on the northeast coast of North America.

 As one of the Thirteen Colonies, New York achieved independence and worked with the others to found the United States.
</item 3>

<item 4>
Proprietary colony
A proprietary colony was a type of British colony mostly in North America and the Caribbean in the 17th century.

 In the British Empire, all land belonged to the ruler, and it was his prerogative to divide.

 Therefore, all colonial properties were partitioned by royal charter into one of four types: proprietary, royal, joint stock, or covenant.

 King Charles II used the proprietary solution to reward allies and focus his own attention on Britain itself.

 He offered his friends colonial charters which facilitated private investment and colonial self-government.

 The charters made the proprietor the effective ruler, albeit one ultimately responsible to English law and the king.

 Charles II gave New Netherland to his younger brother The Duke of York, who named it New York.

 He gave an area to William Penn who named it Pennsylvania.
</item 4>

<item 5>
Province of Pennsylvania
The Province of Pennsylvania, also known as the Pennsylvania Colony, was founded in English North America by William Penn on March 4, 1681 as dictated in a
royal charter granted by King Charles II.

 The name Pennsylvania, which translates roughly as "Penn's Woods", was created by combining the Penn surname (in honor of William's father, Admiral Sir William
Penn) with the Latin word "sylvania", meaning "forest land."

 The Province of Pennsylvania was one of the two major Restoration colonies, the other being the Province of Carolina.

 The proprietary colony's charter remained in the hands of the Penn family until the American Revolution, when the Commonwealth of Pennsylvania was created and
became one of the original thirteen states.
</item 5>

<item 6>
Charter colony
Charter colony is one of three classes of colonial government established in the 17th century English colonies in North America, the other classes being
proprietary colony and royal colony.

 The colonies of Rhode Island, Connecticut, and Massachusetts Bay were charter colonies.

 In a charter colony, Britain granted a charter to the colonial government establishing the rules under which the colony was to be governed.

 The charters of Rhode Island and Connecticut granted the colonists significantly more political liberty than other colonies.

 Rhode Island and Connecticut continued to use their colonial charters as their State constitutions after the American Revolution.
</item 6>

<item 7>
Canada under British rule
Canada first came under British rule with the Treaty of Paris (1763) which ceded New France, of which Canada was a part, to the British Empire.

 Gradually, other territories, colonies, and provinces that were part of British North America would be added to Canada.

 The Royal Proclamation of 1763 enlarged the colony of Canada under the name of the Province of Quebec, which with the Constitutional Act 1791 became known as
The Canadas.

 With the Act of Union 1840 Upper and Lower Canada were joined to become the United Province of Canada.

 Later, with Confederation in 1867, the British maritime colonies of New Brunswick and Nova Scotia were joined with the British colony of Canada to form the
Dominion of Canada, which was subsequently divided into four provinces, Ontario, Quebec, New Brunswick, and Nova Scotia.

 A number of other British colonies, such as Newfoundland and British Columbia, and large territories such as Rupert's Land initially remained outside of the
newly formed federation.

 Over time, the remaining colonies and territories within British North America came under the control of Canada until the current geographic extent of the
country was reached when Newfoundland and Labrador joined Canada in 1949.
</item 7>

<item 8>
British North America Acts
The British North America Acts 1867–1975 are the original names of a series of Acts at the core of the constitution of Canada.

 They were enacted by the Parliament of the United Kingdom and the Parliament of Canada.

 In Canada, some of the Acts were amended or repealed by the Constitution Act, 1982.

 The rest were renamed in Canada as the "Constitution Acts".

 In the United Kingdom, those Acts that were passed by the British Parliament remain under their original names.

 The term "British North America" (BNA) refers to the British colonies in North America.
</item 8>

<item 9>
Stamp Act Congress
The Stamp Act Congress or First Congress of the American Colonies was a meeting held between October 7 and 25, 1765 in New York City, consisting of
representatives from some of the British colonies in North America; it was the first gathering of elected representatives from several of the American colonies
to devise a unified protest against new British taxation.

 Parliament had passed the Stamp Act, which required the use of specially stamped paper for legal documents, playing cards, calendars, newspapers and dice for
virtually all business in the colonies, and was going into effect on November 1.
</item 9>

<item 10>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 10>
=== END ITEM id=bd74ffc993c05278 ===
