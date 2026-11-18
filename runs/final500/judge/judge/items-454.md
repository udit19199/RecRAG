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

=== ITEM id=10bf0959a8816d9c ===
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
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 3>

<item 4>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 4>

<item 5>
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
</item 5>

<item 6>
Edmund Mortimer, 5th Earl of March
Edmund de Mortimer, 5th Earl of March and 7th Earl of Ulster (6 November 1391 – 18 January 1425) was an English nobleman.

 A great-great-grandson of King Edward III of England, he was heir presumptive to King Richard II of England, his first cousin twice removed, when Richard II
was deposed in favour of Henry IV.

 Edmund Mortimer's claim to the throne was the basis of rebellions and plots against Henry IV and his son Henry V, and was later taken up by the House of York
in the Wars of the Roses, though Mortimer himself was an important and loyal vassal of Henry V and Henry VI.

 Edmund Mortimer was the last Earl of March of the Mortimer family.
</item 6>

<item 7>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
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
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 9>

<item 10>
Corps of Military Staff Clerks
The Corps of Military Staff Clerks was an administrative corps of the Canadian Army.

 The Headquarters Corps of Military Staff Clerks was established in 1905.
</item 10>
=== END ITEM id=10bf0959a8816d9c ===

=== ITEM id=fa186657b6409385 ===
Gold sentences and facts:
Sentence 0: Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.
  Fact 0.0: Vytautas Kevin Gerulaitis was born on July 26, 1954.
  Fact 0.1: Vytautas Kevin Gerulaitis died on September 17, 1994.
  Fact 0.2: Vytautas Kevin Gerulaitis was an American professional tennis player.
Sentence 1: Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.
  Fact 1.0: Heather Miriam Watson was born on 19 May 1992.
  Fact 1.1: Heather Miriam Watson is a British tennis player.
  Fact 1.2: Heather Miriam Watson is a Wimbledon Mixed Doubles Champion.

Retrieved items:
<item 1>
1974 Stadthalle Open – Singles
Vitas Gerulaitis won in the final 6–4, 3–6, 6–3, 6–2 against Andrew Pattison.
</item 1>

<item 2>
Vitas Gerulaitis
Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.

 In 1975, Gerulaitis won the men's doubles title at Wimbledon, partnering with Sandy Mayer.

 He won the men's singles title at one of the two Australian Open tournaments held in 1977 (Gerulaitis won the tournament that was held in December, while
Roscoe Tanner won the earlier January tournament).

 Gerulaitis also won two Italian Open titles, in 1977 and 1979, and the WCT Finals in Dallas, in 1978.
</item 2>

<item 3>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 3>

<item 4>
Ernests Gulbis
Ernests Gulbis (] , born 30 August 1988, and nicknamed "The Gull" or "Ernie") is a Latvian professional tennis player.

 In 2008, Gulbis won his first ATP Tour doubles title at the U.S. Men's Clay Court Championships, teaming with Rainer Schüttler, and in 2010 won his first ATP
Tour singles title in the Delray Beach, defeating Ivo Karlović in the final.

 In total, Gulbis has 6 ATP titles to his name and has never lost a final.

 His best performance at a Grand Slam is reaching the semifinals of the 2014 French Open.

 He had previously reached the quarterfinals of the 2008 French Open.

 Gulbis' career-high singles ranking is world No. 10, making him the only Latvian tennis player ever to be ranked inside the top 10 in ATP Singles Ranking.

 He achieved this in June 2014.
</item 4>

<item 5>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 5>

<item 6>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 6>

<item 7>
1977 Italian Open (tennis)
The 1977 Italian Open was a combined men's and women's tennis tournament that was played by men on outdoor clay courts at the Foro Italico in Rome, Italy.

 The men's tournament was part of the Colgate-Palmolive Grand Prix circuit while the women's tournament was part of the Colgate Series.

 The tournament was held from 16 May through 22 May 1977.

 The singles titles were won by eight-seeded Vitas Gerulaitis and fifth-seeded Janet Newberry who earned $21,000 and $6,000 first-prize money respectively.

 Gerulaitis competed despite being contracted to play for the Indiana Loves World Team Tennis (WTT) franchise and was fined $19,000 for failing to play Björn
Borg in the weekend of the final.
</item 7>

<item 8>
1979 World Championship Tennis Finals – Singles
Vitas Gerulaitis was the defending champion but lost in the semifinals to Björn Borg.
</item 8>

<item 9>
Heather Watson
Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.

 She is a former British No.1 and current British No. 2 behind Johanna Konta.
</item 9>

<item 10>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 10>
=== END ITEM id=fa186657b6409385 ===

=== ITEM id=12ae14fae061a7cb ===
Gold sentences and facts:
Sentence 0:  The album was Prince's third to be released that year (along with "Chaos and Disorder" and the soundtrack album of the Spike Lee movie "Girl 6"), which made 1996 one of the most prolific years for material released by Prince.
  Fact 0.0: The album was Prince's third album released in 1996.
  Fact 0.1: One of Prince's other 1996 releases was "Chaos and Disorder".
  Fact 0.2: One of Prince's other 1996 releases was the soundtrack album of the Spike Lee movie "Girl 6".
  Fact 0.3: 1996 was one of the most prolific years for material released by Prince.
Sentence 1:  His production company, 40 Acres and a Mule Filmworks, has produced over 35 films since 1983.
  Fact 1.0: Prince's production company is 40 Acres and a Mule Filmworks.
  Fact 1.1: 40 Acres and a Mule Filmworks has produced over 35 films since 1983.

Retrieved items:
<item 1>
The entries span film, television, documentaries, and film distribution: Philippine series Sine Novela adapts Viva classics; The Doom Generation (1995) is Gregg
Araki’s violent “Teenage Apocalypse” film; Naked Gun 33⅓ concludes its comedy trilogy, while Naked City (1958–63) is a semi-documentary police drama. Night Life
is a zombie comedy; The Runaways portrays the band’s formation and Currie-Jett relationship. Consolidated Film Industries was
</item 1>

<item 2>
Six Cylinder Love
Six Cylinder Love is a lost 1923 silent film comedy produced and distributed by Fox Film and directed by Elmer Clifton.

 It is based on a popular 1921 Broadway play and stars Ernest Truex from the play.

 Also appearing in the film from the Broadway play were Donald Meek and Ralph Sipperly.
</item 2>

<item 3>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 3>

<item 4>
Cannibal boom
Only four other cannibal films were made after 1981 until the fad's conclusion in 1988 with Antonio Climati's "Natura Contro".
</item 4>

<item 5>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 5>

<item 6>
The Human Centipede 2 (Full Sequence)
The Human Centipede 2 (Full Sequence) is a 2011 Dutch black and white exploitation-horror film written, directed, and co-produced by Dutch filmmaker Tom Six.

 The sequel to Six's 2009 film "The Human Centipede (First Sequence)", the film stars Laurence R. Harvey as a mentally impaired English man who watches and
becomes obsessed with the first "Human Centipede" film, and decides to make his own "centipede" consisting of twelve people, including Ashlynn Yennie, an
actress from the first film.
</item 6>

<item 7>
*Girl 6* (1996), Spike Lee’s comedy-drama about a phone-sex operator, stars Theresa Randle and was written by Suzan-Lori Parks. It was Lee’s first film without
his own screenplay, features an all-Prince soundtrack, and screened at Cannes; Quentin Tarantino and Ron Silver cameo. Other profiles cover Issa Rae, creator of
*Awkward Black Girl* and a platform for creators of color; *Family Matters* actress Jaimee Foxworth;
</item 7>

<item 8>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 8>

<item 9>
Still Kicking (film)
Still Kicking: Six Artistic Women of Project Arts & Longevity is a 2006 32-minute documentary film by Pacific Grove filmmaker Greg Young, featuring six Bay Area
women role models over 85 years old who remained artistically active.

 The catalyst for Young's film was Amy Gorman and Frances Kandl's Project Arts & Longevity through which they were exploring the link between longevity and
artistic vitality.

 Along with the film the joint project resulted in a book entitled Aging Artfully.
</item 9>

<item 10>
*Girl 6* is Spike Lee’s 1996 comedy-drama about a phone sex operator, starring Theresa Randle and written by playwright Suzan-Lori Parks. It was Lee’s first
film without a screenplay by him; its soundtrack consists entirely of Prince songs. Screened at Cannes in Un Certain Regard, it features director cameos by
Quentin Tarantino and Ron Silver. Randle’s other films include *Malcolm X*, *Bad Boys*, *Space Jam*, and *Spawn
</item 10>
=== END ITEM id=12ae14fae061a7cb ===

=== ITEM id=283b6d805a7c3053 ===
Gold sentences and facts:
Sentence 0:  His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files" (1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
  Fact 0.0: The subject of this sentence portrayed Bret Maverick in the series Maverick from 1957 to 1962.
  Fact 0.1: The subject of this sentence portrayed Bret Maverick in the series Bret Maverick from 1981 to 1982.
  Fact 0.2: The subject of this sentence portrayed Jim Rockford in the series The Rockford Files from 1974 to 1980.
  Fact 0.3: The subject of this sentence portrayed Jim Egan in the series 8 Simple Rules from 2003 to 2005.
Sentence 1: The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.
  Fact 1.0: The Notebook is a 2004 film.
  Fact 1.1: The Notebook is an American romantic drama film.
  Fact 1.2: The Notebook was directed by Nick Cassavetes.
  Fact 1.3: The Notebook is based on a 1996 novel.
  Fact 1.4: The 1996 novel has the same name as the film The Notebook.
  Fact 1.5: The 1996 novel was written by Nicholas Sparks.

Retrieved items:
<item 1>
Milan (2004 film)
Milan is a 2004 romantic drama film released under Star Cinema, ABS-CBN Film Productions, Inc. in the Philippines.

 It stars Claudine Barretto and Piolo Pascual.

 It is a love story written by Raymond Lee, and under Olivia Lamasan's direction.

 The movie made more than P100 million in the box office.
</item 1>

<item 2>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 2>

<item 3>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 3>

<item 4>
Enamorada (film)
Enamorada ("Enamoured") is a 1946 Mexican drama film directed by Emilio Fernández and starring María Félix and Pedro Armendáriz.

 The film was remade in 1950 as "The Torch" with Armendáriz repeating his role alongside Paulette Goddard who was credited as an associate producer on the film.
</item 4>

<item 5>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 5>

<item 6>
The New Maverick
The New Maverick is a 1978 made-for-TV movie based on the 1957 television series "Maverick", with James Garner as Bret Maverick, Charles Frank as newcomer
cousin Ben Maverick (son of Beau Maverick), Jack Kelly as Bart Maverick, and Susan Sullivan as Poker Alice Ivers.

 Garner had been 29 years old at the beginning of the original series and was 50 while filming "The New Maverick".

 The TV-movie was a pilot for the series "Young Maverick", which featured Frank and only lasted a few episodes.

 Directed by Hy Averback and written by Juanita Bartlett, the movie was filmed while Garner's series "The Rockford Files" was on hiatus.

 Garner would later star in "Bret Maverick", another attempt at a television series revival inspired by this TV-movie, for the 1981-82 season.
</item 6>

<item 7>
Mister Buddwing
Mister Buddwing is a 1966 American film drama directed by Delbert Mann and starring James Garner.
</item 7>

<item 8>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 8>

<item 9>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 9>

<item 10>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 10>
=== END ITEM id=283b6d805a7c3053 ===

=== ITEM id=212c90498810ef3f ===
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
Joseph Saunders led Visa; Gertrude Vanderbilt Whitney was a sculptor and founded the Whitney Museum. Roscoe Channing was a Princeton All-American, Rough Rider,
and mining executive. Fayez Sarofim is a Houston billionaire investor and Texans co-owner; Charles Agemian was a senior banker. The Rockefeller Brothers Fund,
founded in 1940, supports philanthropy. Bankers William Harrison and Walter Shipley led major JPMorgan predecessor mergers. Rockefeller family members include
philanthropists
</item 1>

<item 2>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 2>

<item 3>
JPMorgan Chase
JPMorgan Chase & Co. is a U.S. multinational banking and financial services holding company headquartered in New York City.

 It is the largest bank in the United States, the world's sixth largest bank by total assets, with total assets of US$2.5 trillion, and the world's second most
valuable bank by market capitalization, after ICBC.

 It is a major provider of financial services, and according to "Forbes" magazine is the world's sixth largest public company based upon a composite ranking.

 The hedge fund unit of JPMorgan Chase is the second largest hedge fund in the United States.

 The company was formed in 2000, when Chase Manhattan Corporation merged with J.P. Morgan & Co.
</item 3>

<item 4>
Walter V. Shipley
Walter V. Shipley (born November 2, 1935) was the former Chairman and Chief Executive Officer of Chase Manhattan Bank and, before that, the company with which
it merged Chemical Bank.

 Shipley was named chief executive of Chemical in 1981 and held the position through 1999 and remained at the bank as chairman through January 2000, just prior
to the bank's merger with J.P. Morgan & Co..

 During his 18-year tenure, Shipley oversaw Chemical's mergers with Texas Commerce Bank in 1987, Manufacturers Hanover in 1991 and Chase Manhattan Bank in 1996.
</item 4>

<item 5>
William B. Harrison Jr.
William B. Harrison Jr., born August 12, 1943, in Rocky Mount, North Carolina, is the former CEO and chairman of JPMorgan Chase.

 He attended high school at Virginia Episcopal School, where he was a basketball star.

 He attended the University of North Carolina at Chapel Hill, where he was admitted to the Zeta Psi fraternity.

 Having risen through the ranks of Chemical Bank before succeeding Walter V. Shipley during the Chemical Bank takeover of the Chase Manhattan Corporation, he
and Douglas A. Warner III, then CEO of J.P. Morgan & Co., were the principal architects of the US$30.9 billion Chase and J.P. Morgan & Co. merger of 2000.

 Harrison has been a director of the Firm or a predecessor institution since 1991.

 Harrison is also a director of Merck & Co., Inc.
</item 5>

<item 6>
Hope Aldrich Rockefeller
Hope Aldrich Rockefeller (born May 17, 1938) is an American newspaper publisher.

 She is the eldest daughter of philanthropist John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992).

 Her elder brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).
</item 6>

<item 7>
John D. Rockefeller III
John Davison Rockefeller III (March 21, 1906 – July 10, 1978) was a philanthropist and third-generation member of the prominent Rockefeller family.

 He was the eldest son of philanthropists John D. Rockefeller Jr. and Abby Aldrich Rockefeller.

 His siblings were Abby, Nelson, Laurance, Winthrop, and David.
</item 7>

<item 8>
David Rockefeller
David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.

 He was the oldest living member of the Rockefeller family and family patriarch from August 2004 until his death in March 2017.

 Rockefeller was a son of John D. Rockefeller Jr. and Abby Aldrich Rockefeller, and a grandson of John D. Rockefeller and Laura Spelman Rockefeller.
</item 8>

<item 9>
Rockefeller Brothers Fund
The Rockefeller Brothers Fund (RBF) is a philanthropic foundation created and run by members of the Rockefeller family.

 It was founded in New York City in 1940 as the primary philanthropic vehicle for the five third-generation Rockefeller brothers: John D. Rockefeller III,
Nelson, Laurance, Winthrop and David.

 It is distinct from the Rockefeller Foundation.

 The Rockefellers are an industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the late 19th and
early 20th centuries.
</item 9>

<item 10>
Chase Bank
JPMorgan Chase Bank, N.A., doing business as Chase Bank, is a national bank that constitutes the consumer and commercial banking subsidiary of the U.S.
multinational banking and financial services holding company, JPMorgan Chase.

 The bank was known as Chase Manhattan Bank until it merged with J.P. Morgan & Co. in 2000.

 Chase Manhattan Bank was formed by the merger of the Chase National Bank and The Manhattan Company in 1955.

 The bank has been headquartered in Columbus, Ohio since its merger with Bank One Corporation in 2004.

 The bank acquired the deposits and most assets of Washington Mutual.
</item 10>
=== END ITEM id=212c90498810ef3f ===

=== ITEM id=bfd0c1bac110cf66 ===
Gold sentences and facts:
Sentence 0: The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of the Stones River Campaign in the Western Theater of the American Civil War.
  Fact 0.0: The Battle of Stones River is also called the Second Battle of Murfreesboro.
  Fact 0.1: The Battle of Stones River was fought from December 31, 1862, to January 2, 1863.
  Fact 0.2: The Battle of Stones River was fought in Middle Tennessee.
  Fact 0.3: The Battle of Stones River was the culmination of the Stones River Campaign.
  Fact 0.4: The Stones River Campaign took place in the Western Theater of the American Civil War.
Sentence 1: The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.
  Fact 1.0: The Battle of the Ch'ongch'on River is also known as the Battle of the Ch'ongch'on.
  Fact 1.1: The Battle of the Ch'ongch'on River is also known as the Second Phase Campaign Western Sector.
  Fact 1.2: The Battle of the Ch'ongch'on River was a decisive battle in the Korean War.
  Fact 1.3: The Battle of the Ch'ongch'on River took place from November 25 to December 2, 1950.
  Fact 1.4: The Battle of the Ch'ongch'on River took place along the Ch'ongch'on River Valley.
  Fact 1.5: The Ch'ongch'on River Valley is in the northwestern part of North Korea.

Retrieved items:
<item 1>
Battle of Piqua
The Battle of Piqua, also known as the Battle of Pekowee or Pekowi, was part of the western campaign during the American Revolutionary War.

 Led by General George Rogers Clark, over 1,000 soldiers (among them Daniel Boone and Simon Kenton) crossed the Ohio River near present-day Cincinnati and
burned five Shawnee villages, including Old Chillicothe, along the Little Miami River.

 Peter Loramie's Store, a British trading post-located in what was later Fort Loramie, Ohio in Shelby County, Ohio-, was also burned by Clark's men.

 The Shawnee gradually withdrew during the first few days before finally engaging American forces 7 miles west of Springfield, Ohio on August 8, 1780.

 Joseph Rogers, a cousin of George Rogers Clark, had previously accompanied him to Kentucky and was later captured by the Shawnee near Maysville.

 Despite having been adopted by the tribe, he was killed during the battle while trying to join American forces.
</item 1>

<item 2>
Battle of Stones River
The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of
the Stones River Campaign in the Western Theater of the American Civil War.

 Of the major battles of the Civil War, Stones River had the highest percentage of casualties on both sides.

 Although the battle itself was inconclusive, the Union Army's repulse of two Confederate attacks and the subsequent Confederate withdrawal were a much-needed
boost to Union morale after the defeat at the Battle of Fredericksburg, and it dashed Confederate aspirations for control of Middle Tennessee.
</item 2>

<item 3>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 3>

<item 4>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 4>

<item 5>
Battle of Port Republic
The Battle of Port Republic was fought on June 9, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Port Republic was a fierce contest between two equally determined foes and was the most costly battle fought by Jackson's Army of the Valley during its
campaign.

 Together, the battles of Cross Keys (fought the previous day) and Port Republic were the decisive victories in Jackson's Valley Campaign, forcing the Union
armies to retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 5>

<item 6>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 6>

<item 7>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 7>

<item 8>
Battle of Cross Keys
The Battle of Cross Keys was fought on June 8, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Together, the battles of Cross Keys and Port Republic the following day were the decisive victories in Jackson's Valley Campaign, forcing the Union armies to
retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 8>

<item 9>
Battle of Malvern Hill
The Battle of Malvern Hill, also known as the Battle of Poindexter's Farm, was fought on July 1, 1862 between the Confederate Army of Northern Virginia, led by
Gen. Robert E. Lee, and the Union Army of the Potomac under Maj. Gen. George B. McClellan.

 It was the final battle of the Seven Days Battles during the American Civil War, taking place on a 130 ft elevation of land known as Malvern Hill, near the
Confederate capital of Richmond, Virginia and just one mile (1.6 km) from the James River.

 Including inactive reserves, more than fifty thousand soldiers from each side took part, using more than two hundred pieces of artillery and three warships.
</item 9>

<item 10>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 10>
=== END ITEM id=bfd0c1bac110cf66 ===

=== ITEM id=480ace68902effe4 ===
Gold sentences and facts:
Sentence 0: The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).
  Fact 0.0: The Washington City Canal operated from 1815 until the mid-1850s.
  Fact 0.1: The Washington City Canal operated in Washington, D.C.
  Fact 0.2: The Washington City Canal connected the Anacostia River to Tiber Creek.
  Fact 0.3: The Washington City Canal connected the Anacostia River to the Potomac River.
  Fact 0.4: The Washington City Canal later connected the Anacostia River to the Chesapeake and Ohio Canal (C&O).
  Fact 0.5: The Anacostia River was called the 'Eastern Branch' at that time.
Sentence 1: The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River to Hagerstown, Indiana.
  Fact 1.0: The Whitewater Canal was built between 1836 and 1847.
  Fact 1.1: The Whitewater Canal spanned a distance of seventy-six miles.
  Fact 1.2: The Whitewater Canal stretched from Lawrenceburg, Indiana.
  Fact 1.3: Lawrenceburg, Indiana is on the Ohio River.
  Fact 1.4: The Whitewater Canal stretched to Hagerstown, Indiana.

Retrieved items:
<item 1>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 1>

<item 2>
Duck Creek Aqueduct
Duck Creek Aqueduct, also known as the Metamora Aqueduct and Whitewater Canal Aqueduct, is a historic aqueduct carrying the Whitewater Canal over Duck Creek in
Metamora Township, Franklin County, Indiana.

 It is the only surviving covered wood aqueduct in the United States as such it is a form of covered bridge.

 It is a single-span Burr through truss aqueduct and measures approximately 90 feet long, 25 feet wide, and 25 feet deep.

 The original aqueduct was built between 1839 and 1843.

 The present structure was built in 1846, after the original aqueduct was washed out in a flood.

 It was strengthened in 1868, and repaired in 1901.

 After abandonment and deterioration, the Duck Creek Aqueduct was restored to its present appearance in 1946-1949.
</item 2>

<item 3>
Cincinnati and Whitewater Canal Tunnel
The Cincinnati and Whitewater Canal Tunnel is a historic yet abandoned canal tunnel in the southwestern corner of the U.S. state of Ohio.

 Located within the village of Cleves near Cincinnati, it was constructed in 1837 as part of the Whitewater Canal system.

 Since the canal's closure, it has largely been forgotten, but it has been designated a historic site.
</item 3>

<item 4>
Tiber Creek
Tiber Creek or Tyber Creek is a tributary of the Potomac River in Washington, D.C. In the 19th century it was modified to become part of the Washington City
Canal, and subsequently was enclosed in a tunnel.
</item 4>

<item 5>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 5>

<item 6>
Whitewater Canal
The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River
to Hagerstown, Indiana.
</item 6>

<item 7>
Whitewater River (Great Miami River)
The Whitewater River is a 101 mi southerly flowing right tributary of the Great Miami River in southeastern Indiana and southwestern Ohio in the United States.

 It is formed by the confluence of two forks, the West Fork and East Fork.

 The name is a misnomer, as there is no true white water on the river.

 However, there are many rapids due to the steep gradient present - the river falls an average of 6 ft/mile .

 The gradient rendered upstream navigation impossible, and in the mid-nineteenth century resulted in the construction of the Whitewater Canal paralleling the
river from north of Connersville, Indiana, to the Ohio River.
</item 7>

<item 8>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 8>

<item 9>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 9>

<item 10>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 10>
=== END ITEM id=480ace68902effe4 ===

=== ITEM id=d39a3c3730604fb1 ===
Gold sentences and facts:
Sentence 0: The Australian women's national soccer team represents Australia in international women's soccer at the senior level.
  Fact 0.0: The Australian women's national soccer team represents Australia in international women's soccer.
  Fact 0.1: The Australian women's national soccer team represents Australia at the senior level.
Sentence 1:  The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.
  Fact 1.0: The Australian women's national soccer team's official nickname is the Matildas.
  Fact 1.1: The nickname Matildas comes from the Australian folk song "Waltzing Matilda".
  Fact 1.2: The Australian women's national soccer team was known as the Female Socceroos before 1995.
Sentence 2: "Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
  Fact 2.0: "Waltzing Matilda" is Australia's best-known bush ballad.
  Fact 2.1: "Waltzing Matilda" has been described as the country's "unofficial national anthem."

Retrieved items:
<item 1>
Australia (Gyroscope song)
"Australia" is a song by Australian rock band Gyroscope, from the album "Breed Obsession".

 It was released as the third single from their highly successful album, which peaked at number one on the Australian Albums Chart.

 The song was released to radio in May 2008, and was released as a CD single and digital download on 14 June 2008, along with two acoustic pieces.

 One of those being a live acoustic version of "Australia", the other a live acoustic cover of Chris Isaak's "Wicked Game".
</item 1>

<item 2>
Waltzing Matilda
"Waltzing Matilda" is Australia's best-known bush ballad, and has been described as the country's "unofficial national anthem".
</item 2>

<item 3>
Spencer Park, Queensland
Spencer Park (currently known as Corporate Travel Management Stadium) is a football stadium located in the suburb of Newmarket, north of Brisbane, Queensland,
Australia.

 It is home to National Premier Leagues Queensland team Brisbane City FC.

 The stadium, while officially holding 10,000, seats only 3,000 under the grandstand, the "Gino Merlo Stand".

 The stadium opened in 1963 on the site of an old Brisbane City Council waste disposal site, and underwent a major renovation in 1981 with the construction of
the Gino Merlo Stand.

 The stadium is home to La Rustica restaurant, where traditional Italian foods, including pizza, are served.
</item 3>

<item 4>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 4>

<item 5>
Gambia women's national football team
The Gambia women's national football team represents the Gambia in international football competition.

 The team, however, has not competed in a match recognised by FIFA, the sport's international governing body, despite that organised women's football has been
played in the country since 1998.

 The Gambia has two youth teams, an under-17 side that has competed in FIFA U-17 Women's World Cup qualifiers, and an under-19 side that withdrew from regional
qualifiers for an under-19 World Cup.

 The development of a national team faces challenges similar to those across Africa, although the national football association has four staff members focusing
on women's football.
</item 5>

<item 6>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 6>

<item 7>
Vanuatu national under-23 football team
The Vanuatu national under-23 football team, also known as Vanuatu Cyclone, represents Vanuatu at U23 tournaments.

 The team is considered to be the feeder team for the Vanuatu national football team.

 They are controlled by the Vanuatu Football Federation.

 The team has gained notoriety for thrashing Micronesia's side 46–0 in the 2015 Pacific Games.
</item 7>

<item 8>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 8>

<item 9>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 9>

<item 10>
San Antonio International
San Antonio International was an American soccer club based in San Antonio, Texas that was a member of the Lone Star Soccer Alliance.
</item 10>
=== END ITEM id=d39a3c3730604fb1 ===

=== ITEM id=146dc703eb3544e3 ===
Gold sentences and facts:
Sentence 0: Dieffenbachia is a genus of tropical flowering plants in the family Araceae.
  Fact 0.0: Dieffenbachia is a genus of flowering plants.
  Fact 0.1: Dieffenbachia is a genus of tropical flowering plants.
  Fact 0.2: Dieffenbachia is in the family Araceae.
Sentence 1: Carlina is a genus of flowering plants in the aster family, Asteraceae.
  Fact 1.0: Carlina is a genus of flowering plants.
  Fact 1.1: Carlina is in the aster family.
  Fact 1.2: Carlina is in the family Asteraceae.

Retrieved items:
<item 1>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 1>

<item 2>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 2>

<item 3>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 3>

<item 4>
The entries describe diverse plants and one crab genus. Mascarene Dictyosperma album, the wind-shedding “hurricane palm,” is widely cultivated but nearly
extinct in its native range. Canary Islands Adenocarpus viscosus grows above 1,800 m. Dregea vines occur in Africa and southern Asia; Duranta shrubs and trees
are ornamental, with D. erecta invasive in several regions. Tropical Dieffenbachia are popular houseplants
</item 4>

<item 5>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 5>

<item 6>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 6>

<item 7>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 7>

<item 8>
Carlina
Carlina is a genus of flowering plants in the aster family, Asteraceae.

 It is distributed from Madeira and the Canary Islands across Europe and northern Africa to Siberia and northwestern China.
</item 8>

<item 9>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 9>

<item 10>
The entries describe diverse flowering plants and groups, chiefly from Africa. Ice plants (Aizoaceae) include South African Aloinopsis and Conophytum, and
southern African Conicosia, whose wet-opening capsules release tiny seeds. Other genera span many families and regions: African irises and bulbous amaryllids;
Asian toad lilies and Rodgersia; European and North African Coronilla; and aquatic Alismataceae. Vascular plants are defined by xylem
</item 10>
=== END ITEM id=146dc703eb3544e3 ===

=== ITEM id=4924ce2aaa922257 ===
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
Dann Hume
Dann Hume (born Daniel Benjamin Cobbe, 1 September 1987, Whangaparaoa, New Zealand) is a singer-songwriter and record producer.

 Hume is the youngest of three brothers (with Peter Hume and Jon Hume) who make up the alternative rock band, Evermore since 1999.
</item 1>

<item 2>
Kele Okereke
Kelechukwu "Kele" Okereke (born 13 October 1981), also known by the mononym Kele, is a British musician, best known as the lead singer and rhythm guitarist of
the indie rock band Bloc Party.
</item 2>

<item 3>
Graph query row:
fact: member
person: Jon Hume
relationship: MEMBER_OF
related: Evermore
</item 3>

<item 4>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 4>

<item 5>
Between the Lines (Evermore song)
"Between the Lines" is the first single by Evermore, taken from their third studio album "".

 Evermore's Jon Hume said that "Between the Lines was the first song to come out of a search for a new musical experience as a band."

 It was released as a free download on Evermore's official website on 10 November 2008.
</item 5>

<item 6>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 6>

<item 7>
Evermore (Evermore album)
Evermore is the self-titled international debut album by New Zealand rock band Evermore, which was also released in Australia and New Zealand on 12 March 2010.

 The album is a collection of songs from the band's entire history, as well as three new songs including the single "Underground".
</item 7>

<item 8>
Jon Hume
Jon Hume (born 6 November 1983) is the lead singer of Evermore.

 He is a singer-songwriter, multi-instrumentalist, Record Producer and Studio owner.

 Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
</item 8>

<item 9>
Running (Evermore song)
"Running" is the lead single by alternative rock band Evermore, taken from their second studio album, "Real Life" (July 2006).

 It was written by backing vocalist and drummer, Dann Hume.

 It was issued on 3 June, ahead of the album.

 It was co-produced by Jon with John Alagía (Dave Matthews Band, John Mayer).

 It peaked at No. 4 on their native RIANZ Singles Chart in New Zealand and No. 5 on their adopted country of Australia's ARIA Singles Chart.
</item 9>

<item 10>
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 10>
=== END ITEM id=4924ce2aaa922257 ===

=== ITEM id=0032063a91d98d45 ===
Gold sentences and facts:
Sentence 0: Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.
  Fact 0.0: Emmitt H. Ford was born on December 13, 1943.
  Fact 0.1: Emmitt H. Ford died on November 10, 2014.
  Fact 0.2: Emmitt H. Ford was an American politician.
  Fact 0.3: Emmitt H. Ford was from Tennessee.
Sentence 1:  He was also an uncle of former United States Congressman Harold Ford Jr.
  Fact 1.0: Emmitt H. Ford was an uncle of Harold Ford Jr.
  Fact 1.1: Harold Ford Jr. is a former United States Congressman.
Sentence 2:  He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.
  Fact 2.0: Emmitt H. Ford is the son of Harold Ford Sr.
  Fact 2.1: Harold Ford Sr. is a former Congressman.
  Fact 2.2: Harold Ford Sr. held the same seat for 22 years.

Retrieved items:
<item 1>
Emmitt Ford
Emmitt H. Ford (December 13, 1943 – November 10, 2014) was an American politician from Tennessee.

 He represented the 86th district encompassing Shelby County from 1975–1981.

 He was also an uncle of former United States Congressman Harold Ford Jr.
</item 1>

<item 2>
Doug Ford Jr.
Douglas Bruce Ford Jr. (born November 20, 1964) is a Canadian businessman and politician in Toronto, Ontario.

 Ford was Toronto City Councillor for Ward 2 Etobicoke North in Toronto from 2010 to 2014 at the same time that his brother, Rob Ford (1969–2016), was mayor of
Toronto.

 Their father, Doug Ford Sr., was a Member of Provincial Parliament from 1995 to 1999 and founded Deco Labels and Tags, a printing business operating in Canada
and the United States.

 The business is now owned by Doug Ford Jr., who has served as company president since 2002, and his brother, Randy.
</item 2>

<item 3>
United States Senate election in Tennessee, 2006
The 2006 United States Senate election in Tennessee was held on November 7, 2006.

 The election winner, Bob Corker, served his term between January 3, 2007 and January 3, 2013.

 Corker replaced Republican Bill Frist in the Senate, who retired upon the end of his second term in 2007.

 Corker was the Republican nominee, and the Democratic nominee was Harold Ford, Jr..

 The race between Ford and Corker was one of the most competitive Senate races of 2006, with Corker winning the race by less than three percent of the vote.

 Corker was the only non-incumbent Republican to win a U.S. Senate seat in 2006.

 Since 1994, the Republican Party has held both of Tennessee's Senate seats.
</item 3>

<item 4>
John Y. Brown Jr.
John Young Brown Jr. (born December 28, 1933) is an American politician, entrepreneur, and businessman from the U.S. state of Kentucky.

 He served as the 55th governor of Kentucky from 1979 to 1983, although he may be best known for building Kentucky Fried Chicken (KFC) into a
multimillion-dollar restaurant chain.
</item 4>

<item 5>
John Ford (Tennessee politician)
John N. Ford (born May 3, 1942 in Memphis, Tennessee), is a former Democratic member of the Tennessee State Senate and a member of Tennessee's most prominent
African-American political family.

 He is the older brother of former U.S. Representative Harold Ford, Sr. and the uncle of former Tennessee U.S. Representative and 2006 United States Senate
candidate Harold Ford, Jr. He is a member of the Ford political family from Memphis.
</item 5>

<item 6>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 6>

<item 7>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 7>

<item 8>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 8>

<item 9>
L. R. Ford Jr.
Lester Randolph Ford Jr. (born September 23, 1927 – February 26, 2017) was an American mathematician specializing in network flow problems.

 He was the son of mathematician Lester R. Ford Sr.
</item 9>

<item 10>
Harold Ford Jr.
Harold Eugene Ford Jr. (born May 11, 1970) is an American politician and author who served five terms in the United States House of Representatives as a member
of the Democratic Party from  's 9 congressional district , centered in Memphis, from 1997 to 2007.

 He is the son of former Congressman Harold Ford Sr., who held the same seat for 22 years.

 In 2006, Ford made an unsuccessful bid for the US Senate seat vacated by retiring Bill Frist.

 He is a member of the Ford political family from Memphis.

 Ford was the last chairman of the Democratic Leadership Council (DLC).
</item 10>
=== END ITEM id=0032063a91d98d45 ===

=== ITEM id=a745a8faec32e8a1 ===
Gold sentences and facts:
Sentence 0: Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in 2011.
  Fact 0.0: Pramod Bhasin stepped down as President of Genpact.
  Fact 0.1: Pramod Bhasin stepped down as CEO of Genpact.
  Fact 0.2: Genpact is India's largest business process outsourcing (BPO) company.
  Fact 0.3: Pramod Bhasin became non-executive Vice Chairman in 2011.
Sentence 1: Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.
  Fact 1.0: Genpact is a global business process management and services corporation.
  Fact 1.1: Genpact is an IT corporation.
  Fact 1.2: Genpact has key offices in New York City, United States.

Retrieved items:
<item 1>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 1>

<item 2>
Summit Behavioral Healthcare
Summit Behavioral Healthcare (Summit BHC) is a Brentwood, Tennessee-based behavioral health services company that owns and operates a large network of addiction
treatment centers throughout the United States.

 Originating in 2012, the company was founded by Trey Carter, former CEO of Acadia Healthcare.

 In March 2015, Chicago, IL based investment firm Flexpoint Ford provided Round 1 funding to launch Summit BHC's growth plan.
</item 2>

<item 3>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 3>

<item 4>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 4>

<item 5>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 5>

<item 6>
Mike Long (American businessman)
Mike Long is an American business man, former CEO of several public companies, and currently a founding partner of Sulgrave Partners LLC.

 He served as the President and Chief Executive Officer of Continuum, an Austin, Texas IT consulting company, from 1991 to 1997, having started with Continuum
as a Director in 1983.

 In 1997, Long was named CEO of Healtheon Corporation (now WebMD), succeeding former CEO David Schnell.

 Long oversaw Healtheon's initial public offering, traveling between Europe and the United States to woo investors.

 Long was able to secure the required investment funds, and saw Healtheon's stock price rise from $8 to a high of $120.

 In 2002, Long was recruited to fix the financial struggles of Move, Inc., a company plagued by more than $4 billion in lawsuits and hemorrhaging tens of
millions of dollars a quarter.

 As Chief Executive Officer, Long was able to revive Homestore, Inc., by changing the business model, rebranding the company as Move, Inc. and returning it to
profitability.
</item 6>

<item 7>
Mike Long (American businessman)
and returning it to profitability.

 Touching on his experience of bringing about the initial public offerings of web-based businesses, Long would say that investors needed to be presented "with
an entirely new face every few months," and that "the only way to run one of these Silicon Valley companies was to forget everything you'd learned outside of
Silicon Valley."
</item 7>

<item 8>
Pramod Bhasin
Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in
2011.
</item 8>

<item 9>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 9>

<item 10>
Alex Algard
Alex Algard is an Internet entrepreneur.

 He is the founder and former CEO of CarDomain, the founder and former CEO of Whitepages.com, and the founder and current CEO of Hiya.
</item 10>
=== END ITEM id=a745a8faec32e8a1 ===
