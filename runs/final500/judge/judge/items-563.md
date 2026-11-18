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

=== ITEM id=7553a5855d8ce467 ===
Gold sentences and facts:
Sentence 0: Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.
  Fact 0.0: Pinhead Gunpowder is an American punk rock band.
  Fact 0.1: Pinhead Gunpowder formed in East Bay, California, in 1990.
Sentence 1: Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.
  Fact 1.0: Billie Joe Armstrong was born on February 17, 1972.
  Fact 1.1: Billie Joe Armstrong is an American musician, singer, songwriter and actor.
  Fact 1.2: Billie Joe Armstrong is best known as the lead vocalist, primary songwriter, and guitarist of the punk rock band Green Day.
  Fact 1.3: Billie Joe Armstrong co-founded Green Day with Mike Dirnt.

Retrieved items:
<item 1>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 1>

<item 2>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 2>

<item 3>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 3>

<item 4>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 4>

<item 5>
Green Day related projects
Since 1991, some members of punk band Green Day have branched out past their "main band" and have started other projects with other musicians and have released
full-length albums and several EPs.

 Notable related projects of Green Day include Billie Joe Armstrong's Pinhead Gunpowder (which also featured Green Day's other guitarist Jason White), The
Frustrators in which Mike Dirnt plays bass, and The Network which many speculate has all three members of Green Day, although under stagenames.
</item 5>

<item 6>
Bill Schneider (musician)
William 'Bill' Schneider (born June 17) is an American musician.

 He has played bass in the punk rock band Pinhead Gunpowder since 1990, as well as providing vocals and writing the lyrics to the song "Backyard Flames".

 He has also been in many other bands such as The Influents, Monsula and Uranium 9v, Sawhorse and The Skinflutes.

 Bill was worked as bass tech during the Nimrod era for Mike Dirnt, Green Day's and as guitar tech on their albums "Nimrod" and "Warning", American Idiot, 21st
Century Breakdown, Uno Dos Tres, Revolution Radio and was their day to day manager from 2003-2012 for their "American Idiot" and 21st Century Breakdown tours
through Uno Dos Tres.

 He also owned "Broken Guitars", a guitar shop in Oakland, California with Billie Joe Armstrong.

 Bill Still works on the sidelines with the band as well as in the studio.
</item 6>

<item 7>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 7>

<item 8>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 8>

<item 9>
Carry the Banner
Carry the Banner is the third EP by the Berkeley, California-based punk rock band Pinhead Gunpowder.

 Originally released on 10" vinyl in December 1994 through Too Many Records, the EP was reissued on CD by Lookout Records shortly after as the initial vinyl
pressing sold out quickly.

 It was the group's first release to feature Jason White on guitar/vocals, replacing Sarah Kirsch, who left the band in 1994 due to differences with Billie Joe
Armstrong after his main band Green Day signed to major label Reprise Records.
</item 9>

<item 10>
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 10>
=== END ITEM id=7553a5855d8ce467 ===

=== ITEM id=f3df5e885324d481 ===
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
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 1>

<item 2>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 2>

<item 3>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 3>

<item 4>
Hemigraphis
Hemigraphis is a genus of plants in the family Acanthaceae, consisting of about 30 species native to tropical Asia.

 "Hemigraphis" is sometimes included in the genus "Strobilanthes"
</item 4>

<item 5>
Miltoniopsis
Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.

 This genus comprises 5 species.

 Miltoniopsis's common name is Pansy Orchid.
</item 5>

<item 6>
Miltoniopsis vexillaria
Miltoniopsis vexillaria ("the flag-like "Miltoniopsis"") is a species of epiphytic orchid in the genus Miltoniopsis.
</item 6>

<item 7>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 7>

<item 8>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 8>

<item 9>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 9>

<item 10>
Strobilanthes japonica
Strobilanthes japonica is a flowering herbaceous perennial plant from Asia, one of around 350 plants of the genus "Strobilanthes".

 The 20–50 cm ornamental plant is cultivated in Japan and China, and blooms in autumn with 1.5 cm purple to white funnel-shaped flowers.
</item 10>
=== END ITEM id=f3df5e885324d481 ===

=== ITEM id=c74ab6d4b42ed03e ===
Gold sentences and facts:
Sentence 0:  The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
  Fact 0.0: The documentary explores the history of Sunset Strip.
  Fact 0.1: Sunset Strip is a mile and a half long stretch of road.
  Fact 0.2: Sunset Strip runs through West Hollywood.
Sentence 1:  The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.
  Fact 1.0: The film features some early pioneers of the hardcore punk music scene.
  Fact 1.1: Bad Brains is an early hardcore punk pioneer featured in the film.
  Fact 1.2: Black Flag is an early hardcore punk pioneer featured in the film.
  Fact 1.3: D.O.A. is an early hardcore punk pioneer featured in the film.
  Fact 1.4: Minor Threat is an early hardcore punk pioneer featured in the film.
  Fact 1.5: Minutemen is an early hardcore punk pioneer featured in the film.
  Fact 1.6: SSD is an early hardcore punk pioneer featured in the film.
  Fact 1.7: The film features other early hardcore punk pioneers besides those named.

Retrieved items:
<item 1>
Sunset Strip curfew riots
The Sunset Strip curfew riots, also known as the "hippie riots", were a series of early counterculture-era clashes that took place between police and young
people on the Sunset Strip in Hollywood, California, beginning in the mid-1966 and continuing on and off through the early 1970s.
</item 1>

<item 2>
Riot on Sunset Strip
Riot on Sunset Strip is a 1967 counterculture-era exploitation movie, released by American International Pictures.

 It was filmed and released within four months of the late-1966 Sunset Strip curfew riots.
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
Mayor of the Sunset Strip
Mayor of the Sunset Strip is a 2003 documentary film on the life of Rodney Bingenheimer directed by George Hickenlooper, and produced by Chris Carter.

 In 2011, Craig Hlavaty of the "Houston Press" named "Mayor of the Sunset Strip" at number eight on the paper's list of "The 31 Best Music Documentaries of All
Time".

 The film won the Best Documentary Feature at the 2004 Santa Barbara International Film Festival.
</item 4>

<item 5>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 5>

<item 6>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 6>

<item 7>
Hellfest (American music festival)
Hellfest was an American hardcore punk festival.

 Named for the intense summer heat at the show, the festival originally began featuring hardcore bands before crossing into several genres later on.

 Hellfest was known for its fan-friendly atmosphere, (including a lack of barriers, providing easy access to the stage for stage diving) and for its high
social-awareness, focusing on such causes as animal rights, with many attendees following the straight edge and/or vegan lifestyle.

 It continued as an annual event until it was cancelled in 2005.

 After its cancellation less than thirty-six hours before doors were scheduled to open on the festival's ninth year, some fans, unhappy with the inadequate
processing of refunds, filed small claims court lawsuits against promoter Keith Allen.
</item 7>

<item 8>
Sunset Strip (2012 film)
Sunset Strip is a 2012 documentary directed by Hans Fjellestad, and produced by Tommy Alastra.

 The documentary explores the history of the mile and a half long stretch of road through West Hollywood known as Sunset Strip.
</item 8>

<item 9>
Sunset Strip
Sunset Strip is the mile-and-a-half (2.4 km) stretch of Sunset Boulevard that passes through West Hollywood, California.

 It extends from West Hollywood's eastern border with Hollywood at Crescent Heights Boulevard to its western border with Beverly Hills at Sierra Drive.

 Sunset Strip is probably the best-known portion of Sunset Boulevard, with boutiques, restaurants, rock clubs, and nightclubs.

 It is also known for its array of huge, colorful billboards.
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
=== END ITEM id=c74ab6d4b42ed03e ===

=== ITEM id=421009172155a11b ===
Gold sentences and facts:
Sentence 0: Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.
  Fact 0.0: Christopher Edward Nolan was born 30 July 1970
  Fact 0.1: Christopher Edward Nolan is English-American
  Fact 0.2: Christopher Edward Nolan is a film director
  Fact 0.3: Christopher Edward Nolan is a producer
  Fact 0.4: Christopher Edward Nolan is a screenwriter
Sentence 1: Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.
  Fact 1.0: Paul Joseph Schrader was born July 22, 1946
  Fact 1.1: Paul Joseph Schrader is an American screenwriter
  Fact 1.2: Paul Joseph Schrader is a film director
  Fact 1.3: Paul Joseph Schrader is a film critic

Retrieved items:
<item 1>
Blue Collar (film)
Blue Collar is a 1978 American crime drama film directed by Paul Schrader, in his directorial debut.

 It was written by Schrader and his brother Leonard, and stars Richard Pryor, Harvey Keitel and Yaphet Kotto.
</item 1>

<item 2>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 2>

<item 3>
Christopher Nolan
Christopher Edward Nolan ( ; born 30 July 1970) is an English-American film director, producer, and screenwriter.

 He is one of the highest-grossing directors in history, and among the most successful and acclaimed filmmakers of the 21st century.
</item 3>

<item 4>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 4>

<item 5>
Danny Boyle
Danny Boyle (born 20 October 1956) is an English director, producer, screenwriter and theatre director, known for his work on films including "Shallow Grave",
"Trainspotting", "The Beach", "28 Days Later", "Sunshine", "Slumdog Millionaire", "127 Hours", and "Steve Jobs".

 His debut film "Shallow Grave" won the BAFTA Award for Best British Film.

 Boyle's 2008 film "Slumdog Millionaire" was nominated for ten Academy Awards and won eight, including the Academy Award for Best Director.

 He also won the Golden Globe and BAFTA Award for Best Director.

 Boyle was presented with the Extraordinary Contribution to Filmmaking Award at the 2008 Austin Film Festival, where he also introduced that year's AFF Audience
Award Winner "Slumdog Millionaire".
</item 5>

<item 6>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 6>

<item 7>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 7>

<item 8>
The entries profile filmmakers, producers, performers, and artists: Tim Bedore is a Chicago-born comedian; Rogue One (2016) depicts rebels stealing Death Star
plans; Frederick K. Keller, Peter Guber, Baltasar Kormákur, Steve Rash, Jonathan Kaplan, Kurt Voss, Jim Sharman, Barry Sandler, and Howard Rosenman are
directors or producers. Other highlights include Titan A.E.’s commercial failure; Vito Russo’s documentary; Scott Free’s founding by the
</item 8>

<item 9>
The Walker
The Walker is a 2007 American-British drama film written and directed by Paul Schrader.

 It is an independent production and is the latest installment in Schrader's "night workers" series of films, starting with "Taxi Driver" in 1976, followed by
"American Gigolo" in 1980 and "Light Sleeper" in 1992.
</item 9>

<item 10>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 10>
=== END ITEM id=421009172155a11b ===

=== ITEM id=ab0bc61487f6f257 ===
Gold sentences and facts:
Sentence 0: Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.
  Fact 0.0: Disteghil Sar is also called Distaghil Sar.
  Fact 0.1: Disteghil Sar is the highest mountain in the Hispar Muztagh.
  Fact 0.2: The Hispar Muztagh is a subrange of the Karakoram mountain range.
  Fact 0.3: Disteghil Sar is in Gilgit-Baltistan.
Sentence 1:  The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
  Fact 1.0: Disteghil Sar has a top ridge about 3 km long.
  Fact 1.1: The top ridge of Disteghil Sar is above 7400 m.
  Fact 1.2: Disteghil Sar has three distinct summits.
  Fact 1.3: The (north)west summit of Disteghil Sar is 7885 m high.
  Fact 1.4: The central summit of Disteghil Sar is 7760 m high.
  Fact 1.5: The (south)east summit of Disteghil Sar is 7696 m or 7535 m high.
Sentence 2:  Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
  Fact 2.0: The summit of Disteghil Sar has an elevation of 7295 m.
  Fact 2.1: Disteghil Sar is the highest peak in the Yengisogat subrange of the Karakoram.

Retrieved items:
<item 1>
Yengisogat
The Yengisogat range (音苏盖提 ), also known as the Wesm Mountains, is a Chinese subrange of the Karakoram mountain range.

 It lies north of the Baltoro Muztagh, home of the eight-thousanders of the Karakoram.

 The highest peak is Huangguan Shan, or Crown Peak, 7,265 m (23,835 ft) (also sometimes given as 7,295 m/23,934 ft).
</item 1>

<item 2>
Kezhen Peak
Kezhen Peak, also known as Karpogo Sar, is a mountain in the Karakoram mountain range.

 It is located in the Xinjiang Autonomous Region of China.
</item 2>

<item 3>
Kanjut Sar
Kanjut Sar (Urdu: کنجت سر‎ ) or Kunjudh Sar as pronounced in "Wakhi" is a mountain located in the Hispar Muztagh, a subrange of the Karakoram mountain range.

 Kunjudh Sar in wakhi language mean that which overlooks Kunjudh, or above Kunjudh, while Khujudh is the wakhi name for Lower Hunza.

 It is the 26th highest mountain on Earth and the 11th highest in Pakistan.
</item 3>

<item 4>
The Crown (mountain)
The Crown, also known as Huang Guan Shan and sometimes Crown Peak, is a mountain in the Karakoram mountain range in China.

 It is located in the Xinjiang Uyghur Autonomous Region of China.

 Its summit has an elevation of 7295 m and it is the highest peak in the Yengisogat subrange of the Karakoram.
</item 4>

<item 5>
Momhil Sar
Momhil Sar, or Mumhail Sar as pronounced in Wakhi is at 7343 m above sea level, is the 64th highest mountain peak in the world.

 Mumhail Sar in Wakhi means the mountain that overlooks or is above Grandmother's cattle pen or paddock.

 It is situated in the Hispar Muztagh subrange of the Karakoram range, a few kilometres to the north-west of its parent peak Trivor.
</item 5>

<item 6>
Distaghil Sar
Disteghil Sar or Distaghil Sar (Urdu: ‎ ) is the highest mountain in the Hispar Muztagh, a subrange of the Karakoram mountain range, in Gilgit-Baltistan.

 It is the 19th highest mountain on earth and the 7th highest peak in Pakistan.

 Destghil sar is a Wakhi language word, that means "above the inner ranch."

 The mountain has an about 3 km long top ridge above 7400m with three distinct summits: (north)west 7885m, central 7760 m, and (south)east 7696m or 7535m ().
</item 6>

<item 7>
Passu Sar
Passu Sar (Urdu: ‎ ; or Passu Sar, Passu I) is a mountain peak in the Batura Muztagh, a sub-range of the Karakoram mountain range, located in the Gilgit
District of Gilgit-Baltistan, Pakistan, west of the Hunza Valley.

 It is the high point of the Passu massif, which also includes Passu Diar (or "Passu East", "Pasu II").

 The peak lies on the main ridge of the Batura Muztagh, about 7 km (4 mi) east of Batura Sar.
</item 7>

<item 8>
Yutmaru Sar
Yutmaru Sar is a mountain in the Hispar mountain range, a subrange of the Karakoram.

 At an elevation of 7283 m it is the 88th highest mountain in the world.

 Yutmaru Sar is located in Gilgit-Baltistan, Pakistan.

 It was first climbed in 1980.
</item 8>

<item 9>
Hispar Muztagh
Hispar Muztagh is a sub-range of the Karakoram mountain range.

 It is located in the Gojal region of Gilgit-Baltistan, Pakistan, north of Hispar Glacier, south of Shimshal Valley, and east of the Hunza Valley.

 It is the second highest sub-range of the Karakoram, the highest being the Baltoro Muztagh.

 The highest mountain in the range is Distaghil Sar (7,885m/25,869 ft).
</item 9>

<item 10>
Mingli Sar
Mingli Sar is a mountain located in the Shimshal valley in Gilgit-Baltistan, Pakistan.

 The mountain, located in the Karakoram mountain range, is 6050 meters high and located at the southernmost part of the Pamir mountain range.

 It was first climbed in 1988 by the famous Pakistani climber Nazir Sabir.

 Shimshal lake sits at the base of the mountain.

 China is located to the north, while to the left is the Hindukush mountain range.

 K2 is located on the south eastern side of the valley.
</item 10>
=== END ITEM id=ab0bc61487f6f257 ===

=== ITEM id=c42b6aac17ee627b ===
Gold sentences and facts:
Sentence 0: Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.
  Fact 0.0: Midland F1 Racing was a Formula One constructor.
  Fact 0.1: Midland F1 Racing was a racing team.
  Fact 0.2: Midland F1 Racing is also known as MF1 Racing.
Sentence 1:  In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
  Fact 1.0: In 2007, the team competed as Spyker F1.
  Fact 1.1: In 2008, the team was sold to Indian businessman Vijay Mallya.
  Fact 1.2: In 2008, the team was renamed Force India F1.
Sentence 2: The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.
  Fact 2.0: The Spyker F1 Team was known as the Etihad Aldar Spyker F1 Team for sponsorship reasons.
  Fact 2.1: The Spyker F1 Team was a Formula One team.
  Fact 2.2: The Spyker F1 Team competed in the 2007 Formula One World Championship.
  Fact 2.3: The Spyker F1 Team was created by Spyker Cars after their buyout of the short-lived Midland F1 team.
  Fact 2.4: Midland F1 was formerly known as Jordan Grand Prix.

Retrieved items:
<item 1>
Midland F1 Racing
Midland F1 Racing (also known as MF1 Racing) was a Formula One constructor and racing team.

 It competed in the 2006 Formula One season with drivers Christijan Albers and Tiago Monteiro.

 The team was created by the renaming of Jordan Grand Prix after its purchase by Canadian businessman, and owner of the Midland Group, Alex Shnaider.

 The team was registered as the first Russian Formula One team, reflecting Shnaider's roots, although it continued to be based in the United Kingdom, at
Jordan's Silverstone factory.

 Towards the end of the 2006 season, the team was sold to Spyker Cars N.V.; the team raced in its last three Grands Prix under the official name "Spyker MF1
Racing".

 In 2007, the team competed as Spyker F1, and in 2008 was sold to Indian businessman Vijay Mallya and was renamed Force India F1.
</item 1>

<item 2>
Spyker F8-VII
The Spyker F8-VII (subsequently known as the Force India VJM01) was a Formula One car, constructed by Spyker F1 that competed in the 2007 Formula One World
Championship.

 A "B Specification" car named the Spyker F8-VIIB was launched at the Italian Grand Prix and used for the remainder of the 2007 season.

 The engine of Spyker F8-VII car was Ferrari 056 despite the team opted for 2006-spec engine due to cost reasons.

 For the 2008 World Championship, Force India used a slightly modified version of the F8-VIIB, called the Force India VJM01 named after team owners Vijay
Mallya, Jan Mol and Michiel Mol.

 The VJM01 used 2007-spec Ferrari 056 engines instead of 2008-spec.

 The F8-VII was the only car constructed by Spyker F1 in their own right after their take-over from the struggling Midland F1 team part way through 2006.
</item 2>

<item 3>
Jordan 194
The Jordan 194 was a Formula One car for the 1994 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 Kelvin Burt was named as test driver, but his mileage in the car was limited.

 The engine was a Hart 1035 3.5 V10, a developed version of the engine which had proved promising in 1993.

 The team's main sponsor was Sasol, with additional financial support coming from Arisco, the Irish government and other smaller sponsors.

 With driver aids such as traction control and active suspension banned for the 1994 season, the car used a conventional semi-automatic gearbox and pushrod
double wishbone suspension.

 The simpler technical regulations of 1994 seemed to benefit Jordan and the 194 recaptured the level of performance seen in the team's debut season, 1991.

 Indeed, the cut-down airbox and drooping front wing of the 194 recalled the design features of the 1991 car.
</item 3>

<item 4>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 4>

<item 5>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 5>

<item 6>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 6>

<item 7>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 7>

<item 8>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 8>

<item 9>
Simtek
Simtek (Simulation Technology) was an engineering consultancy firm and Formula One racing team.

 The Formula One (F1) engineering consultancy arm, Simtek Research, was founded in 1989 by Max Mosley and Nick Wirth.

 It originally was involved in many areas of Formula One, including wind tunnel construction and chassis building for third parties.

 Simtek Grand Prix, the racing team, launched in 1993 and competed in the 1994 and 1995 seasons achieving a best result of ninth place.

 With large debts and a lack of sponsorship money, Simtek went into voluntary liquidation in June 1995.
</item 9>

<item 10>
Marussia F1
The Marussia F1 Team (subsequently Manor Marussia F1 Team) was an Anglo-Russian Formula One racing team and constructor which was based in Banbury, Oxfordshire
in the United Kingdom.

 The team was operated by Manor Motorsport (formerly Marussia Manor Racing), which was previously a subsidiary of Marussia Motors, a now defunct sports car
manufacturer which was based in Moscow.

 The team originally started racing in under the "Virgin Racing" name; the following year Virgin adopted Marussia as a title sponsor becoming "Marussia Virgin
Racing" until being fully rebranded as the "Marussia F1 Team" for .
</item 10>
=== END ITEM id=c42b6aac17ee627b ===

=== ITEM id=6c641397051c4298 ===
Gold sentences and facts:
Sentence 0:  For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record when her album "25" took just 29 days to reach the landmark.
  Fact 0.0: For almost 20 years, a record (held by a subject not named in this sentence) stood for selling 2 million copies in the UK in the fastest time.
  Fact 0.1: The record for selling 2 million copies in the UK in the fastest time was set in 42 days.
  Fact 0.2: In 2015, Adele surpassed the record.
  Fact 0.3: Adele's album '25' took just 29 days to reach the 2 million copies landmark.
Sentence 1:  At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
  Fact 1.0: The 51st Grammy Awards were held in 2009.
  Fact 1.1: Adele received the Best New Artist award at the 51st Grammy Awards.
  Fact 1.2: Adele received the Best Female Pop Vocal Performance award at the 51st Grammy Awards.

Retrieved items:
<item 1>
Japan Record Award for Best New Artist
The Japan Record Award for Best New Artist (最優秀新人賞 ) is awarded annually.

 Until the 10th Japan Record Awards it was called New Artist Award, since 11th — Best New Artist Award.

 At present all nominees for the Best New Artist Award are awarded the New Artist Award.
</item 1>

<item 2>
Billboard Year-End
The same method was used for albums based on the "Billboard" 200, and songs appearing on the other charts (e.g. Hot Country Singles).
</item 2>

<item 3>
Billboard Year-End
"Billboard" Year-End charts are a cumulative measure of a single or album's performance in the United States, based upon the "Billboard" magazine charts during
any given chart year.

 "Billboard's" "chart year" runs from the first Billboard "week" of December to the final week in November, but because the Billboard week is dated in advance
of publication, the last calendar week for which sales are counted is usually the third week in November.

 This altered calendar allows for "Billboard" to calculate year-end charts and release them in time for its final print issue in the last week of December.

 Prior to incorporating chart data from Nielsen SoundScan (from 1991), year-end charts were calculated by an inverse-point system based solely on a title's
performance (for example a single appearing on the "Billboard" Hot 100 would be given one point for a week spent at position 100, two points for a week spent at
position ninety-nine, and so forth, up to 100 points for each week spent at number one).

 Other factors including the total weeks a song spent on the chart and at its peak position were calculated into its year-end total.
</item 3>

<item 4>
Robson &amp; Jerome (album)
Robson & Jerome is the debut studio album by British singing duo Robson & Jerome.

 It was released in the UK in 1995 by RCA Records, and reached number one on the UK Albums Chart.

 It was the Christmas number one album of 1995, and was the biggest-selling album of the year, with sales of 2,040,000.

 For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record
when her album "25" took just 29 days to reach the landmark.
</item 4>

<item 5>
List of Billboard 200 number-one albums of 2002
The highest-selling albums and EPs in the United States are ranked in the "Billboard" 200, published by "Billboard" magazine.

 The data are compiled by Nielsen Soundscan based on each album's weekly physical and digital sales.

 25 acts achieved number one albums during this year with artist such as Nelly and Shania Twain who had their albums debut at number one on the chart.

 Rapper Eminem's "The Eminem Show" is the best selling album of 2002 selling over approximately 7.6 million copies by the end of the year.

 It is also the longest running album of 2002 spending six non-consecutive weeks the chart and was known for its first full week of sales debut of 1.322 million
copies which Nielsen SoundScan scanned as the sixth largest sales of all time in its first week.

 Its debut of 1.322 million copies has still not been matched by any album today since except for Taylor Swift's album "1989", which opened with first week
sales of 1.279 million copies.
</item 5>

<item 6>
List of Billboard 200 number-one albums of 2002
The band Creed continued its eight week long run on the chart but is credited as the longest running album 2001.

 Jennifer Lopez earned her second number one album on the charts with "", which became the highest first week sales of a remix album at the time.

 R&B artist Ashanti earned her first number one album with her self-titled debut album "Ashanti", which opened up with first week sales of 503,000 copies in its
first week alone.

 Puff Daddy earned his first number one album since "No Way Out" back in 1997.

 Rapper Jay-Z earned his fifth chart topper with "", which opened up with first week sales of 545,000 copies alone.

 Heavy metal band Disturbed earned its first number one album on the chart with "Believe", which opened up with first week sales of 284,000 copies alone.

 Country music singer Shania Twain's album "Up!

"
</item 6>

<item 7>
Adele
Adele Laurie Blue Adkins (born 5 May 1988) ( ) is an English singer-songwriter.

 After graduating from the BRIT School for Performing Arts and Technology in 2006, Adele was given a recording contract by XL Recordings after a friend posted
her demo on Myspace the same year.

 In 2007, she received the Brit Awards "Critics' Choice" award and won the BBC Sound of 2008 poll.

 Her debut album, "19", was released in 2008 to commercial and critical success.

 It is certified seven times platinum in the UK, and three times platinum in the US.

 The album contains her first song, "Hometown Glory", written when she was 16, which is based on her home suburb of West Norwood in London.

 An appearance she made on "Saturday Night Live" in late 2008 boosted her career in the US.

 At the 51st Grammy Awards in 2009, Adele received the awards for Best New Artist and Best Female Pop Vocal Performance.
</item 7>

<item 8>
MTV Video Music Award for Best New Artist
The MTV Video Music Award for Best New Artist have been given out since the first annual MTV Video Music Awards in 1984.

 From then to 2006, the award was named Best New Artist in a Video.

 In 2007 its name was changed to Best New Artist, as the category underwent a format change to award the artist's body of work for the full year rather than a
specific video.

 For the 2008 ceremony, though, while the award retained its 2007 name, it returned to the format of awarding a specific video rather than the artist's full
body of work.

 The category was later renamed Artist to Watch through 2013 to 2015 while still keeping the format of an award going to a certain video.
</item 8>

<item 9>
List of accolades received by Grand Theft Auto V
Within twenty-four hours of its release, "Grand Theft Auto V" generated more than $815.7 million in worldwide revenue, equating to approximately 11.21 million
copies sold for Take Two.

 Three days after release, the game had surpassed one billion dollars in sales, making it the fastest selling entertainment product in history.
</item 9>

<item 10>
List of Billboard 200 number-one albums of 2002
Country music singer Shania Twain's album "Up!

" opened up with a huge first week sales of 857,000 copies in its first week alone, giving her the recognition of the highest first week sales of her career and
second highest of the year, only behind Eminem's "The Eminem Show" and at the time the fastest selling solo female album ever.

 Nelly's album "Nellyville" opened up with his highest first week sales of his career which logged on with huge sales of 714,000 copies in its first week alone,
which beat his sales of his debut album "Country Grammar", which opened up with first week sales of 235,000 copies.

 Country singer Alan Jackson album "Drive" gave him his first number one album on the chart and opened up with first week sales of 211,000 copies alone.
</item 10>
=== END ITEM id=6c641397051c4298 ===

=== ITEM id=245f9107b46b0052 ===
Gold sentences and facts:
Sentence 0:  The 381-ton vessel was armed with a 12-pounder 12cwt QF gun, a 20mm Oerlikon cannon, a .303-inch Vickers machine gun, and four Type D depth charges, and was commissioned into the RAN on 11 June 1941.
  Fact 0.0: The vessel weighed 381 tons.
  Fact 0.1: The vessel was armed with a 12-pounder 12cwt QF gun.
  Fact 0.2: The vessel was armed with a 20mm Oerlikon cannon.
  Fact 0.3: The vessel was armed with a .303-inch Vickers machine gun.
  Fact 0.4: The vessel was armed with four Type D depth charges.
  Fact 0.5: The vessel was commissioned into the RAN on 11 June 1941.
Sentence 1:  It was in service from before the First World War until the 1960s, with air-cooled versions of it on many Allied World War I fighter aircraft.
  Fact 1.0: This item was in service from before the First World War until the 1960s.
  Fact 1.1: Air-cooled versions of this item were on many Allied World War I fighter aircraft.

Retrieved items:
<item 1>
Vickers machine gun
The Vickers machine gun or Vickers gun is a name primarily used to refer to the water-cooled .303 British (7.7 mm) machine gun produced by Vickers Limited,
originally for the British Army.

 The machine gun typically required a six to eight-man team to operate: one fired, one fed the ammunition, the rest helped to carry the weapon, its ammunition,
and spare parts.

 It was in service from before the First World War until the 1960s, with air-cooled versions of it on many Allied World War I fighter aircraft.
</item 1>

<item 2>
AK-630
The AK-630 is a Soviet and Russian fully automatic naval close-in weapon system based on a six-barreled 30 mm rotary cannon.

 In "630", "6" means 6 barrels and "30" means 30 mm.

 It is mounted in an enclosed automatic turret and directed by radar and television detection and tracking.

 The system's primary purpose is defense against anti-ship missiles and other precision guided weapons.

 However it can also be employed against fixed or rotary wing aircraft, ships and other small craft, coastal targets, and floating mines.

 Once operational, this weapon system was rapidly adopted, with up to 8 units installed in every new Soviet warship (from mine-hunters to aircraft carriers),
and hundreds produced in total.
</item 2>

<item 3>
Pershing II
The Pershing II Weapon System was a solid-fueled two-stage ballistic missile designed and built by Martin Marietta to replace the Pershing 1a Field Artillery
Missile System as the United States Army's primary nuclear-capable theater-level weapon.

 The U.S. Army replaced the Pershing 1a with the Pershing II Weapon System in 1983 while the German Air Force retained Pershing 1a until all Pershings were
eliminated in 1991.

 The U.S. Army Missile Command (MICOM) managed the development and improvements while the Field Artillery Branch deployed the systems and developed tactical
doctrine.
</item 3>

<item 4>
List of sunken aircraft carriers
With the advent of heavier-than-air flight, the aircraft carrier has become a decisive weapon at sea.

 In 1911 aircraft began to be successfully launched and landed on ships with the successful flight of a Curtiss Pusher aboard the USS "Pennsylvania".

 The British Royal Navy pioneered the first aircraft carrier as floatplanes, as flying boats under performed compared to traditional land based aircraft.

 The first true aircraft carrier was the HMS "Argus" , launched in late 1917 with a complement of 20 aircraft, a flight deck of 550 ft and 68 ft wide.

 The last aircraft carrier sunk in wartime was the Japanese carrier "Amagi", in Kure Harbour in July 1945.

 The greatest loss of life was the 2,046 killed on the "Akitsu Maru"—a converted passenger liner with a small flight deck, carrying the Imperial Japanese Army's
64th Infantry Regiment.
</item 4>

<item 5>
HMAS Narani
HMAS "Narani" was an auxiliary minesweeper operated by the Royal Australian Navy (RAN) during World War II.

 "Narani" was requisitioned from the Illawarra & South Coast Steam Navigation Company as auxiliaries.

 The 381-ton vessel was armed with a 12-pounder 12cwt QF gun, a 20mm Oerlikon cannon, a .303-inch Vickers machine gun, and four Type D depth charges, and was
commissioned into the RAN on 11 June 1941.
</item 5>

<item 6>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 6>

<item 7>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 7>

<item 8>
SWARM
The SWARM Remote Weapon System (Stabilised Weapon And Reconnaissance Mount) is a fully armored remote weapon system designed and built by the Thales Group in
Glasgow, Scotland.

 The SWARM system consists of two main assemblies: the Gun Processing and Interface Unit (GPIU), which is operated inside the vehicle, and the external Weapon
and Sensor Platform (WASP).

 It can fire a variety of weapons, and utilize multiple sensors.

 On the US Marine Corps' Gladiator Tactical Unmanned Ground Vehicle (TUGV), it is equipped with a 7.62 mm M240 and day/night sensors.
</item 8>

<item 9>
Weapon system
Weapon System is a United States military term that designated, along with a weapon system number (e.g., WS-110), military experimental (MX) weapons prior to
official naming (e.g., under a military aircraft designation system.

 Preceded by the first Skunk Works program (MX-813 for the Convair XF-92 in 1946), the earliest "WS" designation was the 1954 WS-117L.

 Circa February 1950, an Air Research and Development Command ""study prepared by Maj Gen Gordon P. Saville ... recommended that a 'systems approach' to new
weapons be adopted "[whereby]" development of a weapon "system" required development of support equipment as well as the actual hardware itself.""
</item 9>

<item 10>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 10>
=== END ITEM id=245f9107b46b0052 ===

=== ITEM id=a99d33c10289a4a6 ===
Gold sentences and facts:
Sentence 0: The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.
  Fact 0.0: The University of Arizona is a public research university.
  Fact 0.1: The University of Arizona is in Tucson, Arizona, United States.
  Fact 0.2: The University of Arizona is also referred to as U of A.
  Fact 0.3: The University of Arizona is also referred to as UA.
  Fact 0.4: The University of Arizona is also referred to as Arizona.
Sentence 1: The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.
  Fact 1.0: The Presbyterian University College, Ghana is a partially private and public university.
  Fact 1.1: The Presbyterian University College, Ghana is located at Abetifi-Kwahu.
  Fact 1.2: Abetifi-Kwahu is in the Eastern Region of Ghana.
Sentence 2:  It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.
  Fact 2.0: The Presbyterian University College, Ghana is one of the new universities in Ghana.
  Fact 2.1: The Presbyterian University College, Ghana was granted accreditation by the National Accreditation Board.

Retrieved items:
<item 1>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 1>

<item 2>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 2>

<item 3>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 3>

<item 4>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 4>

<item 5>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 5>

<item 6>
Graph query row:
organization: Presbyterian University College, Ghana
founding_date: 23 November 2003
</item 6>

<item 7>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 7>

<item 8>
Lydia Patterson Institute
Lydia Patterson Institute is a Methodist Christian college-preparatory school located in El Paso, Texas, United States.

 Founded in 1913, it offers programs for Spanish-speaking children, primarily from Juarez, to attend high school in the United States and attend an American
undergraduate university.

 All high school classes are taught in English, and the school is accredited by the Southern Association of Colleges and Schools.
</item 8>

<item 9>
Albanian University in Berat
The University of Berat is a private institution, independent and secular Higher Education, which operates the state fee.

 Licensed by the Council of Ministers, no. 697 dated 18.06.2009, he opened the doors in the academic year 2009- 2010, offering programs in two faculties:
Faculty of Social Sciences Programs in Science Teaching Legal, Political Science and Psychology General Administration and the Faculty of Economics with
Learning programs in Banking and Finance-Business Administration.

 During the first academic year, about 1,000 students attended studies.

 A year later, this new institution offering undergraduate degrees in study programs: Preschool Teacher Education, Primary Education Teacher, English Language,
Nursing, Engineering Computer and Information Technology, thus finalizing the 2010-2011 academic year with the opening of two new faculties: Faculty of Medical
Sciences and the Faculty of Architecture and Inxhinierive.

 Academic Year 2010-2011 finds the University Berat with about 2600 students in the first cycle of study.

 Since the first year of its inception, the institution pointed at the center of academic life, the promotion of intellectual product and melting learning and
research.
</item 9>

<item 10>
Evangelical Presbyterian University College
The Evangelical Presbyterian University College was established in 2008 by the Evangelical Presbyterian Church, Ghana.
</item 10>
=== END ITEM id=a99d33c10289a4a6 ===

=== ITEM id=96f86c537dfb5a87 ===
Gold sentences and facts:
Sentence 0: Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca O'Brien and Lauren Dark.
  Fact 0.0: Mediastan is a documentary film.
  Fact 0.1: Mediastan was released in 2013.
  Fact 0.2: Mediastan is about the 2010 United States diplomatic cables leak.
  Fact 0.3: Mediastan was directed by Johannes Wahlström.
  Fact 0.4: Mediastan was produced by Julian Assange.
  Fact 0.5: Mediastan was produced by Rebecca O'Brien.
  Fact 0.6: Mediastan was produced by Lauren Dark.
Sentence 1: Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he founded in 2006.
  Fact 1.0: Julian Paul Assange was born Julian Paul Hawkins.
  Fact 1.1: Julian Paul Assange was born on 3 July 1971.
  Fact 1.2: Julian Paul Assange is Australian.
  Fact 1.3: Julian Paul Assange is a computer programmer.
  Fact 1.4: Julian Paul Assange is the founder of WikiLeaks.
  Fact 1.5: WikiLeaks is an organisation.
  Fact 1.6: Julian Paul Assange founded WikiLeaks in 2006.

Retrieved items:
<item 1>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 1>

<item 2>
Aryanto Yuniawan
Aryanto Yuniawan (born June 18, 1977 in Magelang; age 40) is Chief Executive Officer of PT Mataram Surya Visi Sinema, a multimedia and animation movie company
located in Jogja Indonesia.

 Ary (his nickname) is best known as a director of Battle of Surabaya, the first 2D animation movie in Indonesia which got some awards i.e. Gold Remi Award USA
2016, GrandPrize Winner of SICAF 2016 in South Korea, Winner of Nice International Filmmaker Festival 2017 in France, Winner of Nioda International Film
Festival in India and others.

 He is also acclaimed as a writer, a producer, and currently working on several articles and books related to his job and hobby.

 He started his career from directing and producing independent animation films including music videos.

 He was trusted to be one of the juries of several film festivals such as Festival Film Indonesia (FFI 2016), Noida International Film Festival (NIFF 2016), and
Seoul International Cartoon and Animation Festival (SICAF 2017).
</item 2>

<item 3>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 3>

<item 4>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 4>

<item 5>
Mira Nair
Mira Nair (born 15 October 1957) is an Indian American filmmaker based in New York City.

 Her production company, Mirabai Films, specializes in films for international audiences on Indian society, whether in the economic, social or cultural spheres.

 Among her best known films are "Mississippi Masala", "The Namesake", the Golden Lion-winning "Monsoon Wedding" and "Salaam Bombay!

", which was nominated for the Academy Award for Best Foreign Language Film.
</item 5>

<item 6>
Mediastan
Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca
O'Brien and Lauren Dark.
</item 6>

<item 7>
Shajoon Kariyal
He debuted as a director with "Rajaputhran" (1996), starring Suresh Gopi, Shobhana and Vikram.

 He has directed many films, including the commercially successful "Thachiledathu Chundan" (1999) and the critically acclaimed "Vadakkumnadhan" (2006).

 After "Vadakkumnadhan", he planned two films, "Raman Police" and "Talkies", but both the projects did not work out.

 In 2012, he directed "Chettayees" which he also co-produced, as one of the five partners of the newly launched production house Thakkaali Films.

 His latest film is "Sir C. P." (2015).
</item 7>

<item 8>
Shamal Sabri
Shamal Sabri (Kurdish: شه‌مال سه‌بری), (born 9 November 1985), is an independent Kurdish award-winning film producer and Duhok International Film Festival
Artistic Director, graduated from Duhok University with BA in English Literature.

.

 He was born in the city of Duhok (Kurdish: دهوك) in Iraqi Kurdistan.

 Apart from being one of the co-founders of Duhok IFF in 2009, he has worked in many other fields of video productions, produced many commercials and music
videos in NV Production and Vin TV.

 Throughout his film carrier, he has contributed in many award-winning films as an assistant director, production manager, assistant producer and line producer.

 His latest contributions were in many films such as "Before Snowfall" by Hisham Zaman, "The Swallow" by Mano Khalil, "Memories on Stone" by Shawkat Amin Korki
and Michael by Kurdo Duski.
</item 8>

<item 9>
Julian Assange
Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he
founded in 2006.

 He has won accolades including the Sam Adams Award and Martha Gellhorn Prize for Journalism.
</item 9>

<item 10>
Faraz Anwar (film)
Faraz Anwar – Documentary is a 2005 documentary film directed and produced by Haroon Sheikh.

 The documentary film follows the experience of Faraz Anwar, lead guitarist and vocalist of the Pakistani progressive rock band Mizraab, in the music industry.

 The documentary was released on November, 2005, by Poison Wood Productions.
</item 10>
=== END ITEM id=96f86c537dfb5a87 ===

=== ITEM id=8b41261adc609c37 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province.
  Fact 0.2: Liaoning province is in the People's Republic of China.
Sentence 1: Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.
  Fact 1.0: Kaiyuan is a county-level city.
  Fact 1.1: Kaiyuan is in the northeast of Liaoning.
  Fact 1.2: Liaoning is in the People's Republic of China.
  Fact 1.3: Kaiyuan borders Jilin for a small section to the north.

Retrieved items:
<item 1>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 1>

<item 2>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 2>

<item 3>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 3>

<item 4>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 4>

<item 5>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 5>

<item 6>
Kaiyuan West Railway Station
Kaiyuan West Railway Station is a railway station of Hada Passenger Railway and located in Liaoning, China
</item 6>

<item 7>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 7>

<item 8>
Wei Junxing
Wei Junxing (; born October 1955) is a Chinese politician who spent most of his career in Liaoning province in northeast China.

 In January 2015, Wei was put under investigation by the Communist Party of China's anti-corruption agency.

 Previously he served as Deputy Secretary-general of Liaoning provincial government and the Party Secretary of Kaiyuan, and briefly as Mayor of Jinzhou.
</item 8>

<item 9>
Du Zheheng
Du Zheheng () (1910–1975) was a People's Republic of China politician.

 He was born in Kaiyuan, Liaoning.

 In 1936, after attending Northeastern University, he went to Xi'an to participate in the Xi'an Incident.

 In 1937, he joined the Communist Party of China.

 During the Second Sino-Japanese War, he was a member of the Eighth Route Army, active in Hebei, Shanxi, Shandong and Henan Provinces.

 He was Secretary of Liaobei Province and vice-chairman of Liaodong (also called Andong Province).

 In 1950, he joined the People's Volunteer Army in the Korean War.

 Upon his return to China, he was agricultural minister for the Northeast Greater Administrative Area.

 In 1954, upon the dissolution of the greater administrative areas, he was made the 1st governor of his home province.
</item 9>

<item 10>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 10>
=== END ITEM id=8b41261adc609c37 ===
