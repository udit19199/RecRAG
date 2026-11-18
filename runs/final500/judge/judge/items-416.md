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

=== ITEM id=85cc761972810b23 ===
Gold sentences and facts:
Sentence 0: August or Auguste Brancart (21 July 1851 - 1894?)
  Fact 0.0: August Brancart is also known as Auguste Brancart.
  Fact 0.1: August Brancart was born on 21 July 1851.
  Fact 0.2: The end date given for August Brancart is 1894, with a question mark.
Sentence 1:  He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
  Fact 1.0: Brancart published work of the Decadent movement.
  Fact 1.1: Brancart published "Monsieur Vénus" by Rachilde.
  Fact 1.2: "Monsieur Vénus" by Rachilde is an example of Decadent movement work.
Sentence 2: The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and artificiality.
  Fact 2.0: The Decadent Movement was an artistic and literary movement.
  Fact 2.1: The Decadent Movement was a late 19th-century movement.
  Fact 2.2: The Decadent Movement was centered in Western Europe.
  Fact 2.3: The Decadent Movement followed an aesthetic ideology of excess.
  Fact 2.4: The Decadent Movement followed an aesthetic ideology of artificiality.

Retrieved items:
<item 1>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 1>

<item 2>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 2>

<item 3>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 3>

<item 4>
Decadent movement
The Decadent Movement was a late 19th-century artistic and literary movement, centered in Western Europe, that followed an aesthetic ideology of excess and
artificiality.

 The visual artist Félicien Rops's body of work and Joris-Karl Huysmans's novel "Against Nature" (1884) are considered the prime examples of the decadent
movement.

 It first flourished in France and then spread throughout Europe and to the United States.

 The movement was characterized by self-disgust, sickness at the world, general skepticism, delight in perversion and employment of crude humor and a belief in
the superiority of human creativity over logic and the natural world.
</item 4>

<item 5>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 5>

<item 6>
Hermann Brunn
Karl Hermann Brunn (1 August 1862 – 20 September 1939) was a German mathematician, known for his work in convex geometry (see Brunn–Minkowski inequality) and in
knot theory.

 Brunnian links are named after him, as his 1892 article "Über Verkettung" included examples of such links.
</item 6>

<item 7>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 7>

<item 8>
Auguste Brancart
August or Auguste Brancart (21 July 1851 - 1894?)

 was a Belgian publisher of pornographic literature, credited with the first publication of "My Secret Life".

 He published translations of English pornography into French and vice versa for English publishers such as Edward Avery.

 He also published work of the Decadent movement such as "Monsieur Vénus" by Rachilde.
</item 8>

<item 9>
Donald Allen
Donald Merriam Allen (Iowa, 1912 – San Francisco, August 29, 2004) was an influential editor, publisher, and translator of contemporary American literature.

 He is perhaps best known for his project "The New American Poetry 1945-1960" (1960), among the several important anthologies of contemporary American
innovative writing he made available to the public.

 Allen began his working life as a Japanese translator within the US military, serving in WWII.

 After his military service, Allen became an editor at Grove Press, where he worked for sixteen years.

 He was one of the first translators of the Romanian-French Absurdist playwright Eugène Ionesco, and Allen's 1958 volume "Four Plays of Eugène Ionesco" helped
to introduce the playwright to American audiences in the 1960s.
</item 9>

<item 10>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 10>
=== END ITEM id=85cc761972810b23 ===

=== ITEM id=2eaf1334fab35a53 ===
Gold sentences and facts:
Sentence 0: "Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.
  Fact 0.0: 'Love Incredible' is a song by Norwegian DJ and record producer Cashmere Cat.
  Fact 0.1: 'Love Incredible' features vocals by Cuban-American singer Camila Cabello.
Sentence 1:  It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
  Fact 1.0: 'Love Incredible' was released on 17 February 2017.
  Fact 1.1: 'Love Incredible' was released through Interscope and Mad Love.
  Fact 1.2: 'Love Incredible' was the third single from the debut album '9' of Cashmere Cat.
Sentence 2:  An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
  Fact 2.0: An imprint of Interscope Geffen A&M Records has Universal Music Group as its parent company.
  Fact 2.1: Universal Music Group is a subsidiary of Vivendi S.A.

Retrieved items:
<item 1>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 1>

<item 2>
Interscope Records
Interscope Records is an American major record label.

 An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
</item 2>

<item 3>
Manimal Vinyl
Manimal Vinyl is a Calabasas based record label founded in 2006 by experimental musician and fashion stylist, Paul Beahan.

 They were originally known as a psychedelic folk label that makes semi-annual tribute records for worldwide charities, having eventually switched to dance
music and progressive underground.

 They are the label who has release debut records from act ranging from Warpaint to Bat for Lashes and exclusive singles from Moby, Duran Duran, Carla Bruni,
Edward Sharpe and the Magnetic Zeros.

 In 2008, they founded the annual Manimal Festival in Pioneertown, California which ran from 2008-2010.

 The festival has featured line-ups with Edward Sharpe and the Magnetic Zeros, Warpaint, Ariel Pink and papercranes.

 Manimal has offices in Los Angeles and New York City.
</item 3>

<item 4>
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
</item 4>

<item 5>
Range Life (song)
The single was not commercially released in the USA; it was issued by the band's UK label at the time, Big Cat.

 Both B-sides are outtakes from the "Crooked Rain, Crooked Rain" sessions and are included on the 2004 of that album.

 This song was one of many to be included in the group's greatest hits album "".
</item 5>

<item 6>
Incredible Connection
Incredible Connection (Pty) Ltd is the largest consumer electronics and IT retailer in South Africa.

 On 15 December 1998, Incredible Connection was acquired by Connection Group Holdings Limited and now operates as subsidiary of that company, itself a
subsidiary of JD Group Limited.

 As of 2005, Incredible Connection has 34 retail outlets in South Africa, and in 2002 it was reported to hold 45% of the IT retail market in the country.
</item 6>

<item 7>
Inamorata (album)
Inamorata is the 18th album by the country rock band Poco.

 Featuring guest spots by former members Timothy B. Schmitt, Richie Furay and George Grantham (the only member missing from the classic years line up was Jim
Messina), this would be the last album that the band would record for Atlantic Records.

 After this the original line up (Richie Furay, Rusty Young, Jim Messina, Randy Meisner and George Grantham) would reform for the 1989 release "Legacy".
</item 7>

<item 8>
American Record Corporation
American Record Corporation (ARC), also referred to as American Record Company, American Recording Corporation, or (erroneously) as ARC Records, was an American
record company.

 It resulted from the merger in 1929 of three companies: the Cameo Record Corporation (which owned Cameo, Lincoln and Romeo Records), the Pathé Phonograph and
Radio Corporation (which owned Actuelle, Pathé, and Perfect), and the Plaza Music Company (which owned Banner, Domino, Jewel, Oriole, and Regal).
</item 8>

<item 9>
Kinetic Records
The label was distributed by Reprise/Warner Bros.

 Records from its inception until 2001, when Kinetic switched distribution to BMG.

 The latter eventually absorbed the label during the merger with Sony Music.
</item 9>

<item 10>
Love Incredible
"Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.

 It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
</item 10>
=== END ITEM id=2eaf1334fab35a53 ===

=== ITEM id=133b88b3bb803ea5 ===
Gold sentences and facts:
Sentence 0:  The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover "Come Together".
  Fact 0.0: The tour accompanied the release of the piano-based album 'In the Running'.
  Fact 0.1: The tour featured pared-back versions of his hits.
  Fact 0.2: The tour featured pared-back versions of his album tracks.
  Fact 0.3: The tour featured a pared-back version of The Beatles cover 'Come Together'.
Sentence 1:  The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
  Fact 1.0: The song reached the top of the charts in the United States.
  Fact 1.1: The song peaked at No. 4 in the United Kingdom.

Retrieved items:
<item 1>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 1>

<item 2>
Live Acoustic America
Live Acoustic America is an acoustic live recording of classic Howard Jones songs, released in 1996.

 It was recorded in Los Angeles to a sell out crowd and features Carol Steele on percussion.

 The acoustic tour travelled the world, with Jones playing venues in Europe, the USA and Jamaica.

 The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover
"Come Together".
</item 2>

<item 3>
Range Life (song)
The single was not commercially released in the USA; it was issued by the band's UK label at the time, Big Cat.

 Both B-sides are outtakes from the "Crooked Rain, Crooked Rain" sessions and are included on the 2004 of that album.

 This song was one of many to be included in the group's greatest hits album "".
</item 3>

<item 4>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 4>

<item 5>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 5>

<item 6>
On Top of Old Smoky
"On Top of Old Smokey" is a traditional folk song of the United States.

 As recorded by The Weavers, the song reached the pop music charts in 1951.

 It is catalogued as Roud Folk Song Index No. 414.

 In one version the first verse is the following; for more on the words see below.
</item 6>

<item 7>
Baby, Stop Crying
"Baby, Stop Crying" is a song written by Bob Dylan, released in the summer of 1978 as a single and in a longer album version on "Street Legal".

 It was performed live only during his big-band worldwide tour of 1978.

 The song charted at #13 in the UK and was a top-ten song in much of Europe, although it failed to chart in the United States.
</item 7>

<item 8>
Beatles (song)
"Beatles", written by Sven Olov Bagge and Claes Bure, was the song that, performed by Swedish dansband Forbes, won the Swedish Melodifestivalen 1977.

 The song's lyrics are about the popular 1960s British rock group the Beatles and include the phrase "Yeah, yeah, yeah" as an homage to the Beatles song "She
Loves You".

 Forbes took the song to the United Kingdom for the Eurovision Song Contest 1977, with Anders Berglund as conductor.

 The audience cheered at the song, not unexpectedly since the contest took place in the United Kingdom, but the jury had another opinion.

 The song finished last, with only two points, from West Germany.
</item 8>

<item 9>
Afrodiziak
In 1984 and 1985 they performed on Madness's albums "Keep Moving" and "Mad Not Mad" respectively, also appearing in the video for the Madness single "Sweetest
Girl".

 Also in 1985, they performed backing vocals on the first two songs on Howard Jones' second album, "Dream Into Action", both of which ("Things Can Only Get
Better" and "Life in One Day") became worldwide hit singles; later that year, they went on tour as part of Jones' band.

 Other acts the group sang backup for include Japan, Ian Gillan, Aztec Camera, Maxi Priest, Aswad, Sam Brown, and Julia Fordham.

 Besides "The Tube", Afrodiziak also appeared on "Top of the Pops", "The Old Grey Whistle Test", "Solid Gold", and "American Bandstand".

 They performed live at the Montreux Jazz Festival, Live Aid Japan, Red Nose Day of Charity, and the first Free Nelson Mandela Concert.
</item 9>

<item 10>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 10>
=== END ITEM id=133b88b3bb803ea5 ===

=== ITEM id=e6910fb50bde2c09 ===
Gold sentences and facts:
Sentence 0: Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.
  Fact 0.0: Xingcheng is a county-level city of southwest Liaoning province, China.
  Fact 0.1: Xingcheng's former name was Ningyuan (宁远).
  Fact 0.2: Xingcheng has a population of approximately 140,000 urban inhabitants.
  Fact 0.3: Xingcheng is located on Liaodong Bay, the northern coast of the Bohai Sea.
Sentence 1:  It covers an area of 2046 square kilometers and has a population of about one million.
  Fact 1.0: Xingcheng covers an area of 2046 square kilometers.
  Fact 1.1: Xingcheng has a population of about one million.

Retrieved items:
<item 1>
Shanxi–Henan–Shandong Railway
The line can accommodate train speeds of up to 120 km/h .

 Cities along route include Lüliang, Liulin, Hongdong, Changzhi and Pingshun in Shanxi; Linzhou, Anyang, Tangyin, Puyang, Hebi, Fan County and Taiqian County in
Henan; Liangshan County, Dongping, Ningyang, Xintai, Laiwu, Yiyuan, Yishui, Ju County, Junan and Rizhao in Shandong.

 The line was built from 2010 to 2014 and operations began on December 30, 2014.
</item 1>

<item 2>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 2>

<item 3>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 3>

<item 4>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 4>

<item 5>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 5>

<item 6>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 6>

<item 7>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 7>

<item 8>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 8>

<item 9>
Linzhou, Henan
Linzhou (), formerly Lin County (林县), is a county-level city in Anyang, Henan, China.

 Adjacent to Shanxi Province and Hebei Province, it is located in the north of Henan Province and at the eastern foot of the Taihang Mountains.

 It covers an area of 2046 square kilometers and has a population of about one million.
</item 9>

<item 10>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 10>
=== END ITEM id=e6910fb50bde2c09 ===

=== ITEM id=1f70bd1f12f6f493 ===
Gold sentences and facts:
Sentence 0: "Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the majority of their members quit and formed Faith No More.
  Fact 0.0: "Quiet in Heaven" / "Song of Liberty" is a double A-side single.
  Fact 0.1: "Quiet in Heaven" / "Song of Liberty" is the only official release by Faith No Man.
  Fact 0.2: Faith No Man is stylised as Faith•No Man•.
  Fact 0.3: The single was released before the majority of Faith No Man's members quit.
  Fact 0.4: After the majority of Faith No Man's members quit, they formed Faith No More.
Sentence 1:  Before settling on their current name in 1982, the band performed under the names Sharp Young Men and later Faith No Man.
  Fact 1.0: The band settled on its current name in 1982.
  Fact 1.1: Before settling on its current name in 1982, the band performed under the name Sharp Young Men.
  Fact 1.2: The band later performed under the name Faith No Man.

Retrieved items:
<item 1>
Kiss Me Quiet (song)
"Kiss Me Quiet" is a song recorded by Canadian country singer Jess Moskaluke for her second extended play of the same name (2015).

 It was written by Kelly Archer, Corey Crowder, and Emily Shackelton and was produced by Crowder.

 "Kiss Me Quiet" was released August 28, 2015 as the EP's lead single.

 The song was Moskaluke's first solo top 10 on the Canada Country chart.
</item 1>

<item 2>
Faith No More
Faith No More (sometimes abbreviated as FNM) is an American rock band from San Francisco, California, formed in 1979.

 Before settling on their current name in 1982, the band performed under the names Sharp Young Men and later Faith No Man.

 Bassist Billy Gould and drummer Mike Bordin are the longest remaining members of the band, having been involved with Faith No More since its inception.

 The band underwent several lineup changes early in their career, along with some major changes later on.

 The current lineup of Faith No More consists of Gould, Bordin, keyboardist Roddy Bottum, guitarist Jon Hudson and vocalist Mike Patton.
</item 2>

<item 3>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 3>

<item 4>
Hoarse &amp; Roaring
Hoarse & Roaring is the debut full-length album by neofolk band Parlor Hawk, released in 2010 on Northplatte Records.

 The album was produced by Joshua James and featured Neon Trees bassist Branden Campbell on each track (except "Saddest Song") as well as Fictionist member
Stuart Maxfield on the tracks "14 Years" and "Home".

 It was mixed by Todd Burke (Ben Harper, Jack Johnson) and mastered by Reuben Cohen (Edward Sharpe & The Magnetic Zeros, Bruno Mars).
</item 4>

<item 5>
Jefferson's Tree of Liberty
Jefferson's Tree of Liberty is the tenth album by Jefferson Starship, released on September 2, 2008.

 It is the band's first studio album since 1999's "Windows of Heaven".

 The new album includes cover songs from Irish, American, English, and Latin-American traditions.

 The title is a reference to Thomas Jefferson's quotation, "The tree of liberty must be refreshed from time to time with the blood of patriots and tyrants."

 The idea began in 2003 as "The Cuba Project", which was to include classic protest and folk songs recorded in Cuba.

 In 2008 the album was finally recorded but in California.

 About half of the songs planned for The Cuba Project were used on the final cut, with other songs coming from Jefferson Starship's previous repertoire and
another project band member Paul Kantner had planned called "On the Threshold of Fire."

 The promotional tour for the album began in late June with shows at Larkspur, California followed by tours in the US and Europe before the album's release, and
continued through December 2008 with a further tour in the US and a tour in Japan.
</item 5>

<item 6>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 6>

<item 7>
Quiet in Heaven/Song of Liberty
"Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the
majority of their members quit and formed Faith No More.

 The songs have not appeared in any of the band's compilation albums and have not been performed live since Mike Morris left the band.
</item 7>

<item 8>
Corcovado (song)
"Corcovado" (known in English as "Quiet Nights of Quiet Stars") is a bossa nova song written by Antônio Carlos Jobim in 1960.

 An English lyric was later written by Gene Lees.

 The Portuguese title refers to the Corcovado mountain in Rio de Janeiro.

 Andy Williams recorded the song with English lyrics, reaching #92 in the "Billboard" Hot 100 and #18 in the Hot Adult Contemporary Tracks chart in 1965.

 Also receiving air-play, contemporaneously with Andy Williams' recording of "Quiet Nights," was Kitty Kallen's version.

 Her album, titled "Quiet Nights," was released by 20th Century-Fox Records in 1964.
</item 8>

<item 9>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 9>

<item 10>
List of songs recorded by Faith No More
Faith No More, a San Francisco-based band, have recorded approximately 100 songs over the course of their career; this includes material from six studio albums,
one live album, and numerous B-side tracks and out-takes.

 Faith No More were founded in 1981 as Sharp Young Men, and changed their name to Faith No Man before releasing the 1982 double A-side single "Quiet in Heaven"
/ "Song of Liberty".

 Upon assuming the name Faith No More, the band's first two full-length albums, "We Care a Lot" and "Introduce Yourself", were driven mostly by new vocalist
Chuck Mosley and the "metallic guitar" of Jim Martin, blending elements of rap and heavy metal music.

 Mosley was later replaced by Mr. Bungle vocalist Mike Patton, who added lyrics to the already-written music for 1989's "The Real Thing".

 Their next studio album, "Angel Dust", moved away from the band's rap-influenced sound to experiment with different musical genres; a trend which became much
more pronounced on the 1995 album "King for a Day... Fool for a Lifetime".
</item 10>
=== END ITEM id=1f70bd1f12f6f493 ===

=== ITEM id=3e938af60f8a34ce ===
Gold sentences and facts:
Sentence 0: Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in 1995.
  Fact 0.0: Haunted is the second album by American singer-songwriter Poe.
  Fact 0.1: Haunted was released in 2000.
  Fact 0.2: Haunted was released after a five-year hiatus from Poe's debut album 'Hello', which came out in 1995.
Sentence 1:  The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
  Fact 1.0: Haunted was self-produced.
  Fact 1.1: Haunted was created as a tribute to Poe's father.
  Fact 1.2: Haunted was created as a counterpart to the novel 'House of Leaves' by Poe's brother Mark Z. Danielewski.
Sentence 2:  A bestseller, it has been translated into a number of languages, and is followed by a companion piece, "The Whalestoe Letters".
  Fact 2.0: Haunted is a bestseller.
  Fact 2.1: Haunted has been translated into a number of languages.
  Fact 2.2: Haunted is followed by a companion piece, 'The Whalestoe Letters'.

Retrieved items:
<item 1>
The Padlock
The Padlock is a two-act 'afterpiece' opera by Charles Dibdin.

 The text was by Isaac Bickerstaffe.

 It debuted in 1768 at the Drury Lane Theatre in London as a companion piece to "The Earl of Warwick".

 It partnered other plays before a run of six performances in tandem with "The Fatal Discovery" by John Home.

 "The Padlock" was a success, largely due to Dibdin's portrayal of Mungo, a blackface caricature of a black servant from the West Indies.

 The company took the production to the United States the next year, where a portrayal by Lewis Hallam, Jr. as Mungo met with even greater accolades.

 The libretto was first published in London in 1768 (?)

 and in Dublin in 1775.

 The play remained in regular circulation in the U.S. as late as 1843.

 It was revived by the Old Vic Company in London and on tour in the UK in 1979 in a new orchestration by Don Fraser and played in a double-bill with Garrick's
"Miss In Her Teens".
</item 1>

<item 2>
James Poe
James Poe (October 4, 1921 – January 24, 1980) was an American film and television screenwriter.

 He is best known for his work on the movies "Around the World in 80 Days" for which he jointly won an Academy Award in 1956, "Cat on a Hot Tin Roof", "Summer
and Smoke", "Lilies of the Field", and "They Shoot Horses, Don't They?

".
</item 2>

<item 3>
Uncle Samuel
Uncle Samuel is a one-act comic opera with a libretto by Arthur Law and music by George Grossmith.

 It was first produced at the Opera Comique on 3 May 1881 to 8 October 1881, as companion piece to "Patience".

 The piece also toured from December 1887 to June 1888 as a companion piece to "H.M.S. Pinafore".
</item 3>

<item 4>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 4>

<item 5>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 5>

<item 6>
The Spectre Knight
The Spectre Knight is a one-act "fanciful operetta" with a libretto by James Albery and music by Alfred Cellier.

 It was first performed on 9 February 1878 at the Opera Comique by the Comedy Opera Company as a companion piece to "The Sorcerer".

 The piece continued to run until 23 March 1878 and was revived by the company from 28 May 1878 to 10 August 1878 as a companion piece to "H.M.S. Pinafore".

 The piece had a run in New York in 1880 and was toured in Britain and America.
</item 6>

<item 7>
A Predicament
"A Predicament" is a humorous short story by Edgar Allan Poe, usually combined with its companion piece "How to Write a Blackwood Article."

 It was originally titled "The Scythe of Time".

 The paired stories parody the Gothic sensation tale, popular in England and America since the early 19th century.
</item 7>

<item 8>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 8>

<item 9>
Haunted (Poe album)
Haunted is the second album by American singer-songwriter Poe, released in 2000 (see 2000 in music) after a five-year hiatus from her debut album "Hello" in
1995.

 The self-produced album was created as a tribute to her father, and counterpart to her brother Mark Z. Danielewski's novel "House of Leaves".
</item 9>

<item 10>
Incest: From a Journal of Love
Incest: From a Journal of Love: The Unexpurgated Diary of Anaïs Nin (1932–1934) is a 1992 non-fiction book by Anaïs Nin.

 It is a continuation of the diary entries first published in "Henry and June: From the Unexpurgated Diary of Anaïs Nin".

 It features Nin's relationships with writer Henry Miller, his wife June Miller, the psychoanalyst Otto Rank, her father Joaquín Nin, and her husband Hugh
Parker Guiler.

 She also copied some of her correspondence with these people into her diary.

 Much of this book was written in English, although those of her letters which were originally written in French and Spanish were translated.

 Most of this diary takes place in France, particularly Clichy, Paris and Louveciennes.
</item 10>
=== END ITEM id=3e938af60f8a34ce ===

=== ITEM id=e44aa3ed91eeedcc ===
Gold sentences and facts:
Sentence 0: Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.
  Fact 0.0: Edward Sharpe and the Magnetic Zeros is an American folk rock band.
  Fact 0.1: Edward Sharpe and the Magnetic Zeros was formed in Los Angeles, California.
  Fact 0.2: Edward Sharpe and the Magnetic Zeros was formed in 2007.
Sentence 1: Louis XIV is an American rock band from San Diego, California.
  Fact 1.0: Louis XIV is an American rock band.
  Fact 1.1: Louis XIV is from San Diego, California.
Sentence 2:  The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.
  Fact 2.0: The band released four EPs between 2003 and 2007.
  Fact 2.1: The band released three albums between 2003 and 2008.
  Fact 2.2: Two of the band's three albums, the latter two, were distributed by Atlantic Records.

Retrieved items:
<item 1>
Alexander (Alex Ebert album)
Alexander is the first solo album by American musician Alex Ebert, lead singer of Edward Sharpe and the Magnetic Zeros and Ima Robot.

 It was released on March 1, 2011 under the Community Records label.

 The song "Truth" was featured in the season 4 premiere of AMC's "Breaking Bad" titled "Box Cutter".
</item 1>

<item 2>
Orpheo McCord
Orpheo McCord (born 1979) is an American drummer and percussionist.

 He is an original member of Edward Sharpe and the Magnetic Zeros, and a former member of The Fall, Fool's Gold and Cass McCombs.
</item 2>

<item 3>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 3>

<item 4>
Edward Sharpe and the Magnetic Zeros
Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.

 The group is led by lead singer Alex Ebert.

 The band's name is based on a story Ebert wrote in his youth, about a messianic figure named Edward Sharpe.

 Drawing from roots rock, folk, gospel, and psychedelic music, the band's image and sound evoke the hippie movement of the 1960s and 1970s.

 The group's first show was played July 18, 2007 at The Troubadour in West Hollywood, California.

 Their first studio album, "Up from Below", was released on July 7, 2009 on Community Records and featured the popular single "Home".

 The group released their second full-length album, "Here", on May 29, 2012, and third album, "Edward Sharpe and the Magnetic Zeros", on July 23, 2013.

 Their fourth studio album, "PersonA", was released in April 2016.
</item 4>

<item 5>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 5>

<item 6>
PersonA
PersonA (pronounced "Persona") is the fourth studio album by the band Edward Sharpe and the Magnetic Zeros.

 It was released on April 15, 2016.
</item 6>

<item 7>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
</item 7>

<item 8>
Louis XIV (band)
Louis XIV is an American rock band from San Diego, California.

 The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.

 The band broke up in 2009, but in 2013, in an interview with The Reno Dispatch, Jason Hill confirmed that the band had decided to reunite.
</item 8>

<item 9>
Beasts of Bourbon
Beasts of Bourbon are an Australian alternative rock, blues rock band formed in August 1983, with James Baker on drums (ex-Hoodoo Gurus), Spencer P. Jones on
guitar (The Johnnys), Tex Perkins on vocals (Dum Dums), Kim Salmon on guitar and Boris Sujdovic on bass guitar (both ex-The Scientists).

 Except for mainstays, Jones and Perkins, the line-up has changed as the group splintered and reformed several times.

 Their debut album, "The Axeman's Jazz" (July 1984), was the best selling Australian alternative rock album for 1984.

 Their debut single, "Psycho", is a cover version of the Leon Payne original, and was the best selling Australian alternative rock single for that year.

 However the group disbanded by mid-1985 and each member pursued other musical projects.

 They reformed in 1987 and issued a second album, "Sour Mash", in December 1988 on Red Eye Records.

 According to rock music historian, Ian McFarlane, it "virtually redefined the parameters of guitar-based rock'n'roll.
</item 9>

<item 10>
Edward Sharpe and the Magnetic Zeros (album)
Edward Sharpe and the Magnetic Zeros is the third album by Edward Sharpe and the Magnetic Zeros.

 It was released on July 23, 2013 in North America and was released on July 29, 2013 around the world through Vagrant Records, Rough Trade Records and Communion
Records.

 Frontman Alex Ebert stated that "These songs mean everything to me - It's the rawest, most liberated, most rambunctious stuff we've done."
</item 10>
=== END ITEM id=e44aa3ed91eeedcc ===

=== ITEM id=081b2faa736bf29e ===
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
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 1>

<item 2>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 2>

<item 3>
Wilmington International Airport
Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated
Wrightsboro, Cape Fear Township, New Hanover County.
</item 3>

<item 4>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 4>

<item 5>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 5>

<item 6>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 6>

<item 7>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 7>

<item 8>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 8>

<item 9>
Pensacola International Airport
Pensacola International Airport (IATA: PNS, ICAO: KPNS, FAA LID: PNS) , formerly Pensacola Gulf Coast Regional Airport and Pensacola Regional Airport (Hagler
Field), is a public use airport three nautical miles (6 km) northeast of the central business district of Pensacola, in Escambia County, Florida, United States.

 It is owned by the City of Pensacola.

 Despite the name, this airport does not offer direct international flights.

 This airport is one of the five major airports in North Florida, others being: Northwest Florida Beaches International Airport, Destin-Fort Walton Beach
Airport Tallahassee International Airport, and Jacksonville International Airport.
</item 9>

<item 10>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 10>
=== END ITEM id=081b2faa736bf29e ===

=== ITEM id=0bb9bff763612d19 ===
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
Various Positions (film)
Various Positions is a 2002 film by Vancouver, BC, lawyer and filmmaker Ori Kowarsky, starring Carly Pope and Tygh Runyan.

 "Various Positions" won the 2002 Prix de Montréal at the Montreal World Film Festival.

 Although the film takes its title from an album by (and Ira Nadel's biography of) Leonard Cohen, the subject of the film is not Cohen, nor does he have any
affiliation with the work.
</item 1>

<item 2>
Various Positions Tour
The Various Positions Tour was a concert tour by Leonard Cohen, in support of his album "Various Positions", released in 1984 (Canada) and 1985 in Europe and
the United States.
</item 2>

<item 3>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 3>

<item 4>
Love Can't Wait (Nick Carter song)
"Love Can't Wait" is a song by American singer-songwriter Nick Carter, released as the lead Canadian single from his second studio album, "I'm Taking Off".

 The single was released on June 21, 2011.

 The song only appears on the Canadian version of the album, and has not been released in any other territories.

 The music video for the song features Carter performing the song with the backdrop of a motorway system.

 The music video also featured Canadian pop star Shawn Desman, who co-wrote the song.

 The single peaked at #2 on the Canadian Singles Chart, based entirely on download sales.

 Carter ran a competition for amateur mixers to remix the track, with the winning remixer earning $1500 and their remix appearing on Carter's remix album, "I'm
Taking Off: Relaunched and Remixed".
</item 4>

<item 5>
Guillaume Morissette
Guillaume Morissette (born 1984) is a Canadian fiction writer and poet based in Montreal, Quebec.

 His work has frequently been associated with the Alt Lit movement, with Dazed & Confused magazine describing him as "Canada's Alt Lit poster boy."

 He has published stories, poems and essays online and in print, in venues such as "Maisonneuve", "Little Brother", "Broken Pencil", "Shabby Doll House" and
"Thought Catalog", and was listed as one of CBC Books' "Writers to Watch" for 2014.
</item 5>

<item 6>
Of Montreal
of Montreal is an American experimental pop band from Athens, Georgia.

 It was founded by frontman Kevin Barnes in 1996, named after a failed romance between Barnes and a woman "of Montreal."

 The band is identified as part of the Elephant 6 collective.

 Throughout its existence, of Montreal's musical style has evolved considerably and drawn inspiration from numerous 1960s psychedelic pop acts.
</item 6>

<item 7>
Daddy's Hands (band)
Daddy's Hands was a Canadian experimental rock band formed in Victoria, British Columbia, Canada, by singer/guitarist Dave Wenger, bassist Emily Bauslaugh,
drummer Jonah Fortune, and saxophonist Jonathan Pollard.

 Dave Wenger and Emily Bauslaugh had been involved in the British Columbian hardcore/emo scene before forming the band, being in bands M Blanket and Ache Hour
Credo, and the band Floragore respectively.

 Daddy's Hands has released two demo tapes, one EP and two full-length albums.

 The band gigged extensively in the 90s, even playing a radio show with Nardwuar.

 Despite achieving little commercial success, the band has been referred to as highly influential in the West Coast music scene.

 Lead singer Wenger, highly regarded by some critics, has been referred to as Montreal's Peter Laughner.
</item 7>

<item 8>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 8>

<item 9>
For Those Who Think Young (album)
For Those Who Think Young, appearing on the album cover as (for those who think young) and originally to be entitled for those who think jung, was the third
album by Canadian new wave band Rough Trade; it was released in 1981 (True North TN-48 in Canada; Boardwalk NB-33261-1, US; Big Time, UK; CBS 85385, The
Netherlands).

 It climbed to #9 in Canada on the "RPM" Top 50 Albums Chart on November 7, 1981 (putting it at #1 on the CANCON Chart listing), and held the position for three
weeks, dropping out of the Top 50 after sixteen weeks on February 6 of the following year.

 It was certified gold in Canada by the CRIA on November 1, 1981.

 The single "All Touch" gave the band its biggest commercial success, reaching #12 in Canada on the "RPM" Top 50 Singles Chart (#2 on the CANCON Chart) and #58
on the U.S. ""Billboard" Hot 100".
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
=== END ITEM id=0bb9bff763612d19 ===

=== ITEM id=44be19f925c1b2ac ===
Gold sentences and facts:
Sentence 0: John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
  Fact 0.0: John André lived from 2 May 1750 to 2 October 1780.
  Fact 0.1: John André was a British Army officer.
  Fact 0.2: John André was hanged as a spy by the Continental Army during the American Revolutionary War.
  Fact 0.3: John André was hanged for assisting Benedict Arnold's attempted surrender of the fort at West Point, New York, to the British.
Sentence 1: Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American Continental Army, and later defected to the British Army.
  Fact 1.0: Benedict Arnold lived from January 14, 1741 (Old Style: January 3, 1740) to June 14, 1801.
  Fact 1.1: Benedict Arnold was a general during the American Revolutionary War.
  Fact 1.2: Benedict Arnold fought for the American Continental Army.
  Fact 1.3: Benedict Arnold later defected to the British Army.

Retrieved items:
<item 1>
William Farrar Smith
William Farrar Smith (February 17, 1824 – February 28, 1903), known as ‘Baldy’ Smith, was a Union general in the American Civil War, notable for attracting the
extremes of glory and blame.

 He was praised for his gallantry in the Seven Days Battles and the Battle of Antietam, but was demoted for insubordination after the disastrous defeat at the
Battle of Fredericksburg.

 As chief engineer of the Army of the Cumberland, he achieved recognition by restoring a supply line that saved that army from starvation and surrender, known
the “Cracker Line”, that helped Union troops to success in the Chattanooga Campaign in the autumn of 1863.

 Leading the first operation against Petersburg, Smith’s hesitation, possibly illness-related, cost the Union a prime opportunity for a quick end to the war,
and he was relieved of command.
</item 1>

<item 2>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 2>

<item 3>
Siege of Yorktown
The Siege of Yorktown, also known as the Battle of Yorktown, the Surrender at Yorktown, German Battle or the Siege of Little York, ending on October 19, 1781,
at Yorktown, Virginia, was a decisive victory by a combined force of American Continental Army troops led by General George Washington and French Army troops
led by the Comte de Rochambeau over a British Army commanded by British peer and Lieutenant General Charles Cornwallis.

 The culmination of the Yorktown campaign, the siege proved to be the last major land battle of the American Revolutionary War in the North American theater, as
the surrender by Cornwallis, and the capture of both him and his army, prompted the British government to negotiate an end to the conflict.

 The battle boosted faltering American morale and revived French enthusiasm for the war, as well as undermining popular support for the conflict in Great
Britain.
</item 3>

<item 4>
List of Continental Army units
The Continental Army was the national army of first the Thirteen Colonies, and then the independent United States, during the American Revolutionary War,
established by a resolution of the Congress on June 14, 1775, three days before the Battle of Bunker Hill, where it saw its first action under that title.

.

 The Continental Congress took a number of steps in the spring of 1775 to create the army in response to the Battles of Lexington and Concord in April and the
seizure of Fort Ticonderoga in May.

 The units composing the Continental Army changed frequently, especially in the first two years of the war.

 From 1777 to the close of the war, the organization of the Continental Army became progressively more systematic and sophisticated.

 The Continental Army that served at Yorktown in 1781 bore very little resemblance to the Continental Army that blockaded Boston in 1775.
</item 4>

<item 5>
John André
John André (2 May 1750 – 2 October 1780) was a British Army officer hanged as a spy by the Continental Army during the American Revolutionary War for assisting
Benedict Arnold's attempted surrender of the fort at West Point, New York to the British.
</item 5>

<item 6>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 6>

<item 7>
Benjamin Logan
Benjamin Logan (c.1742 – December 11, 1802) was an American pioneer, soldier, and politician from Shelby County, Kentucky.

 As colonel of the Kentucky County, Virginia militia during the American Revolutionary War, he was second-in-command of all the militia in Kentucky.

 He was also a leader in Kentucky's efforts to become a state.

 His brother, John Logan, was the first state treasurer of Kentucky.
</item 7>

<item 8>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 8>

<item 9>
Battle of Germantown
The Battle of Germantown was a major engagement in the Philadelphia campaign of the American Revolutionary War.

 It was fought on October 4, 1777, at Germantown, Pennsylvania, between the British Army led by Sir William Howe, and the American Continental Army, with the
2nd Canadian Regiment, under George Washington.
</item 9>

<item 10>
Benjamin Lincoln
Benjamin Lincoln (January 24, 1733 (O.S. January 13, 1732) – May 9, 1810) was an American army officer.

 He served as a major general in the Continental Army during the American Revolutionary War.

 Lincoln is notable for being involved in three major surrenders during the war: his participation in the Battles of Saratoga (sustaining a wound shortly
afterward) contributed to John Burgoyne's surrender of a British army, he oversaw the largest American surrender of the war at the 1780 Siege of Charleston,
and, as George Washington's second in command, he formally accepted the British surrender at Yorktown.
</item 10>
=== END ITEM id=44be19f925c1b2ac ===

=== ITEM id=94620d56e9f27a96 ===
Gold sentences and facts:
Sentence 0: Park Sung-woong (born January 9, 1973) is a South Korean actor.
  Fact 0.0: Park Sung-woong was born on January 9, 1973.
  Fact 0.1: Park Sung-woong is a South Korean actor.
Sentence 1:  Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a serial killer in "The Deal" (2015).
  Fact 1.0: Park Sung-woong made his acting debut in 'No. 3' in 1997.
  Fact 1.1: Park Sung-woong has starred in several movies and television series.
  Fact 1.2: Park Sung-woong notably starred as a gangster in 'New World' (2013).
  Fact 1.3: Park Sung-woong notably starred as a serial killer in 'The Deal' (2015).
Sentence 2: The Deal (; lit.
  Fact 2.0: The Deal's literal translation is given in the next sentence, after the 'lit.' marker.
Sentence 3:  "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
  Fact 3.0: The literal translation of The Deal is 'Murder Request'.
  Fact 3.1: The Deal is a 2015 South Korean crime thriller film.
  Fact 3.2: The Deal was directed by Son Yong-ho.

Retrieved items:
<item 1>
Hidden Identity (TV series)
Hidden Identity () is a 2015 South Korean television series starring Kim Bum, Park Sung-woong, Yoon So-yi and Lee Won-jong.

 It aired on tvN from June 16 to August 4, 2015 on Mondays and Tuesdays at 23:00 for 16 episodes.
</item 1>

<item 2>
Park Sung-woong
Park Sung-woong (born January 9, 1973) is a South Korean actor.

 Following his acting debut in "No. 3" in 1997, Park has starred in several movies and television series, notably as a gangster in "New World" (2013) and a
serial killer in "The Deal" (2015).
</item 2>

<item 3>
Fatal Intuition
Fatal Intuition (; lit.

 "It's Him") is a 2015 South Korean crime thriller film written and directed by Yun Jun-hyeong.
</item 3>

<item 4>
The Swindlers (2017 film)
The Swindlers is an upcoming South Korean crime drama film directed by Jang Chang-won.

 The film stars Hyun Bin, Yoo Ji-tae, Bae Seong-woo, Park Sung-woong, Nana and Ahn Se-ha.
</item 4>

<item 5>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 5>

<item 6>
The Boys Who Cried Wolf
The Boys Who Cried Wolf (), also known as The Shepherd, is a 2015 South Korean crime thriller drama film.

 Written and directed by Kim Jin-hwang in his first feature-length for his Korean Academy of Film Arts (KAFA)'s final year undergraduate film project, it
depicts how a former stage actor becomes involved in a murder case.
</item 6>

<item 7>
Office (2015 South Korean film)
Office (오피스 ) is 2015 South Korean slasher-thriller film directed by Hong Won-chan and starring Go Ah-sung and Park Sung-woong.

 It is about a detective trying to figure out why a mild-mannered man has killed his family and is targeting his co-workers.

 The film premiered at the Midnight Screenings section of the 2015 Cannes Film Festival.
</item 7>

<item 8>
A Violent Prosecutor
A Violent Prosecutor is a 2016 South Korean crime film directed by Lee Il-hyung, produced by Guk Su Ran and starring Hwang Jung-min, Kang Dong-won, Lee Sung-min
and Park Sung-woong.

 It was released in South Korea on February 3, 2016 by Showbox.
</item 8>

<item 9>
A Company Man
A Company Man () is a 2012 South Korean action thriller film starring So Ji-sub, Lee Mi-yeon, Kwak Do-won and Kim Dong-jun.

 It is about a hitman who find himself targeted by his ex-employers after he falls in love with a single mother and quits his job.
</item 9>

<item 10>
Deep Trap
Deep Trap () (previously known as Exchange) is a 2015 South Korean crime thriller film starring Ma Dong-seok, Jo Han-sun and Kim Min-kyung, and directed by Kwon
Hyung-jin.

 Based on an actual SNS crime, it depicts the terror experienced by a married couple while on a trip to an isolated island.

 It won Best Film Award in the Orient Express section at the Fantasporto in 2016.
</item 10>
=== END ITEM id=94620d56e9f27a96 ===
