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

=== ITEM id=ef1ab2edceeffd3f ===
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
</item 1>

<item 2>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 2>

<item 3>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 3>

<item 4>
Takes a Little Time (Total Contrast song)
"Takes a Little Time" is a 1985 dance hit by British, techno-soul, duo, Total Contrast.

 Their debut American release went to number one on the U.S. dance chart for one week.

 "Takes a Little Time" did not make the Hot 100 but, peaked at number eighty on the R&B singles chart.
</item 4>

<item 5>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 5>

<item 6>
Breathing Your Love
"Breathing Your Love" is a song by the Swedish singer Darin featuring vocals by singer Kat DeLuna and the first single from "Flashback".

 The song co-written by Darin with RedOne, Bilal Hajji and Novel was released to radio stations and as a digital download in Sweden on 8 October 2008.

 The single was released in Finland and is also the first single from Darin to be released in the United Kingdom in January 2010.

 In October 2009 Swedish astronaut Christer Fuglesang took 2 copies of the single into space with him, as his daughter is a Darin fan, a photo of the disc can
be seen with earth in the distance can be found on the official Darin website.
</item 6>

<item 7>
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
</item 7>

<item 8>
The entries cover albums and songs across pop, dance, and rock: Marc Lavoine’s successful 2005 album *L’Heure d’été*; Axelle Red’s 1993 signature hit
“Sensualité”; Lynsey de Paul’s 1994 album *Just a Little Time*, mixing re-recorded hits and new songs; Total Contrast’s 1985 U.S. dance-chart No. 1 “Takes a
Little Time”; Amy Grant’s 1997
</item 8>

<item 9>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 9>

<item 10>
Howard Jones’s 1996 *Live Acoustic America* captures a sold-out Los Angeles show from a worldwide tour, featuring pared-back hits, album tracks, “Come
Together,” and percussionist Carol Steele. Jerry Garcia (1942–1995), Grateful Dead guitarist and vocalist, composed “Sugaree” with Robert Hunter for his 1972
solo album. *Garcia Plays Dylan* compiles 1973–95 live Dylan covers. Bluegrass musician Sandy Rothman
</item 10>
=== END ITEM id=ef1ab2edceeffd3f ===

=== ITEM id=58047c35702b1f2d ===
Gold sentences and facts:
Sentence 0: Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.
  Fact 0.0: Marc Predka was born April 21, 1975.
  Fact 0.1: Marc Predka is better known by his stage name Tha Trademarc.
  Fact 0.2: Marc Predka is an American hip hop artist.
Sentence 1:  He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance theme "The Time Is Now".
  Fact 1.0: Marc Predka first came to prominence when he collaborated with his first cousin, John Cena.
  Fact 1.1: John Cena is a WWE wrestler.
  Fact 1.2: Marc Predka and John Cena collaborated on the 2005 album 'You Can't See Me'.
  Fact 1.3: The album 'You Can't See Me' includes Cena's entrance theme 'The Time Is Now'.
Sentence 2: "The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.
  Fact 2.0: 'The Time Is Now' is a song by John Cena.
  Fact 2.1: John Cena is a professional wrestler.
  Fact 2.2: John Cena is an actor.
  Fact 2.3: John Cena is a hip hop musician.

Retrieved items:
<item 1>
The Preytells
The Preytells were an Australian indie rock band from Perth, formed in 2004 by Jessica Bennett (a.k.a. Audrey Tell) on lead guitar and backing vocals, Simon
Okely (a.k.a. Will Tell) on guitar and lead vocals, Jaclyn Pearson on drums and percussion and Cameron Stewart on bass guitar and backing vocals.

 Their sole album, "Flood Songs/June Songs", was issued in September 2009; before they disbanded later that year.

 Their single, "Shout!"

 ( May 2008), was nominated for WAM Song of the Year in the Pop category in 2008; while, "Lord Hold My Hand", was nominated for the same category in 2009.
</item 1>

<item 2>
Zach Crowell
Zach Crowell is a Grammy Award-winning songwriter and record producer, who was born and raised in Nashville, Tennessee.

 Crowell is involved in a joint publishing deal with Warner/Chappell Music, Combustion Music and Taperoom Music.
</item 2>

<item 3>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 3>

<item 4>
Rebuild (song)
"Rebuild" is a song written by Jon Foreman of Switchfoot and Matt Thiessen of Relient K.

 The song was written in conjunction with the bands' 2007 Appetite for Construction Tour, featuring members from all three bands on tour, Switchfoot, Relient K,
and Ruth.

 Thiessen sang second vocals, and Dustin Ruth of Ruth played harmonica.
</item 4>

<item 5>
Anthem for the Underdog
Anthem for the Underdog is the third album by the rock band 12 Stones.

 It was released on August 14, 2007.

 The first single "Lie to Me", was made available on 12 Stones' MySpace and official website along with the second single "Anthem For The Underdog", which was
used in the movie "Never Back Down".

 Both singles charted in the top 30 on the Mainstream Rock Tracks chart, while "It Was You" charted in the top 10 on Christian Rock charts.

 The third single, "Adrenaline", was the theme song for the Met-Rx World's Strongest Man competition in 2007.

 "Anthem for the Underdog" debuted at No. 53 on the "Billboard" 200, and stayed on the chart for four weeks.
</item 5>

<item 6>
Brand New Love Affair (song)
"Brand New Love Affair," sometimes alternatively listed as "Brand New Love Affair (Parts I and II)", is a song written by James Pankow for the group Chicago and
recorded for their album Chicago VIII.

 The song peaked at #61 on the charts.

 Guitarist Terry Kath sings the first half while bassist Peter Cetera sings the second half.

 Keyboardist Robert Lamm played the distinctive Fender Rhodes electric piano on the song—the intro particularly showcases its lush vibrato bell-like sound.
</item 6>

<item 7>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 7>

<item 8>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 8>

<item 9>
The Time Is Now (John Cena song)
"The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.

 It is the current entrance theme Cena uses as he makes his entrance to the ring.

 Currently wrestling for WWE, Cena recorded the song in 2005 for his debut studio album, "You Can't See Me".
</item 9>

<item 10>
Tha Trademarc
Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.

 He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance
theme "The Time Is Now".

 He also appeared in the music videos of the songs, "Bad Bad Man" and "Right Now" with Cena.

 Prior to the album, Trademarc appeared on Cena's first WWE DVD "Word Life".

 He now works as a classroom specialist for the Peace Corps of America.
</item 10>
=== END ITEM id=58047c35702b1f2d ===

=== ITEM id=0106be915d672598 ===
Gold sentences and facts:
Sentence 0: Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.
  Fact 0.0: Psychopsis is abbreviated Psychp in horticultural trade.
  Fact 0.1: Psychopsis is a genus of 5 known species of orchids.
  Fact 0.2: Psychopsis orchids are native to northern South America.
  Fact 0.3: Psychopsis orchids are native to Central America.
  Fact 0.4: Psychopsis orchids are native to Trinidad.
Sentence 1: Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.
  Fact 1.0: Aubrieta is commonly known as Aubretia.
  Fact 1.1: Aubrieta is a genus of about 12 species of flowering plants.
  Fact 1.2: Aubrieta is in the cabbage family Brassicaceae.

Retrieved items:
<item 1>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 1>

<item 2>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
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
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Aubrieta
Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.

 The genus is named after Claude Aubriet, a French flower-painter.

 It originates from southern Europe east to central Asia but is now a common garden escape throughout Europe.

 It is a low, spreading plant, hardy, evergreen and perennial, with small violet, pink or white flowers, and inhabits rocks and banks.

 It prefers light, well-drained soil, is tolerant of a wide pH range, and can grow in partial shade or full sun.
</item 6>

<item 7>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 7>

<item 8>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 8>

<item 9>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 9>

<item 10>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 10>
=== END ITEM id=0106be915d672598 ===

=== ITEM id=5fb3f2b852238a7a ===
Gold sentences and facts:
Sentence 0:  It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black matipo", "Pittosporum tenuifolium", is not related to "Myrsine").
  Fact 0.0: Myrsine contains about 200 species.
  Fact 0.1: Myrsine includes several notable radiations.
  Fact 0.2: The matipo of New Zealand is one of the notable radiations within Myrsine.
  Fact 0.3: The kōlea of Hawaiʻi is one of the notable radiations within Myrsine.
  Fact 0.4: The New Zealand 'black matipo', Pittosporum tenuifolium, is not related to Myrsine.
Sentence 1: Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.
  Fact 1.0: Storax or snowbell are the common names of Styrax.
  Fact 1.1: Styrax is a genus of about 130 species.
  Fact 1.2: Styrax species are large shrubs or small trees.
  Fact 1.3: Styrax belongs to the family Styracaceae.
  Fact 1.4: Styrax is mostly native to warm temperate to tropical regions of the Northern Hemisphere.
  Fact 1.5: The majority of Styrax species are in eastern and southeastern Asia.
  Fact 1.6: Styrax also crosses the equator in South America.

Retrieved items:
<item 1>
Styrax litseoides
Styrax litseoides is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is endemic to Vietnam.
</item 1>

<item 2>
Pterostyrax
Pterostyrax, the epaulette tree, is a small genus of four species of deciduous large shrubs or small trees in the family Styracaceae, native to eastern Asia in
China and Japan.

 They grow 4 - tall, with alternate, simple ovate leaves 6 - long and 4 - broad.

 The flowers are white, produced in dense panicles 8 - long.

 The fruit is an oblong dry drupe, with longitudinal ribs or narrow wings (the wings are absent in the related genus "Styrax", whence the name "Pterostyrax",
"winged styrax").
</item 2>

<item 3>
Styrax crotonoides
Styrax crotonoides is a species of plant in the genus "Styrax" and family Styracaceae.

 It is found in Peninsular Malaysia and Singapore.
</item 3>

<item 4>
Styrax vilcabambae
Styrax vilcabambae (syn.

 "Pamphilia vilcabambae" D.R.Simpson) is a species of flowering plant in the genus "Styrax" and the family Styracaceae.

 It is endemic to Peru.
</item 4>

<item 5>
Styrax
Storax or snowbell is the common names of Styrax, a genus of about 130 species of large shrubs or small trees in the family Styracaceae, mostly native to warm
temperate to tropical regions of the Northern Hemisphere, with the majority in eastern and southeastern Asia, but also crossing the equator in South America.

 The resin obtained from the tree is called storax or benzoin (not to be confused with the "Liquidambar" storax balsam).
</item 5>

<item 6>
Styrax fraserensis
Styrax fraserensis is a species of flowering plant in the genus "Styrax" and family Styracaceae.

 It is a tree endemic to Peninsular Malaysia.

 It is threatened by habitat loss.
</item 6>

<item 7>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 7>

<item 8>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 8>

<item 9>
Rapanea
Rapanea is a genus of plant in family Primulaceae.

 It has often been placed in synonymy with "Myrsine", and many species have been moved to "Myrsine".
</item 9>

<item 10>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 10>
=== END ITEM id=5fb3f2b852238a7a ===

=== ITEM id=507c3cec9977033a ===
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
Oldham County, KY (2010: 60,316; seat La Grange), named for William Oldham, became fully wet in 2015 after limited restaurant sales began in 2005. Casey County
(15,955; Liberty), formed 1806, is dry, Appalachian, wholly in the Knobs, and hosts an Apple Festival. Virginia’s Kentucky County (1776–c.1780) had its seat at
Harrodstown. Piney Woods School,
</item 1>

<item 2>
Wheeling Island
Wheeling Island is the most populated island in the Ohio River.

 It lies within the city of Wheeling in Ohio County, West Virginia, in the United States.

 The 2000 census showed a resident population of 3,142 people on the island, which has a land area of 1.514 km² (374 acres).

 Neville Island, PA is larger (3.4 km²) but has fewer people (1,232).
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
Oldham County, Kentucky
Oldham County is a county located in the commonwealth of Kentucky.

 As of the 2010 census, the population was 60,316.

 Its county seat is La Grange.

 The county is named for Colonel William Oldham.

 Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants
that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.

 After a vote in late 2015; Oldham county has become a completely wet county.
</item 4>

<item 5>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 5>

<item 6>
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 6>

<item 7>
Lake Louisvilla, Louisville
Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.

 It is located between Westport Road in Louisville and KY 22 in Oldham County.

 Lake Louisvilla was developed in the 1920s as a summer resort for people living in the city of Louisville.

 The state of Kentucky drained the lake in the late 1980s due to safety concerns regarding the stability of a dam.
</item 7>

<item 8>
Casey County, Kentucky
Casey County is a county located in the U.S. Commonwealth of Kentucky.

 As of the 2010 census, the population was 15,955.

 Its county seat is Liberty.

 The county was formed in 1806 from the western part of Lincoln County and named for Colonel William Casey, a pioneer settler who moved his family to Kentucky
in 1779.

 It is the only Kentucky county entirely in the Knobs region.

 Casey County is home to annual Casey County Apple Festival, and is a prohibition or dry county.

 It is considered part of the Appalachian region of Kentucky.
</item 8>

<item 9>
Lake Louisvilla began as a 1920s resort; its lake was drained over dam-safety concerns. Miami’s Flagami is bounded by canals, roads and Flagler Street.
Broward’s unified fire numbering, adopted in 1990 after confusion at a 1988 fire, supports coordinated radio communication. Seoul has complex official and guide
road numbers. Long Island’s 75-mile outer barrier includes Fire and Jones Beach islands. Other routes include Boca Raton’s SR 800, Bay
</item 9>

<item 10>
Kentucky County, Virginia
Kentucky County (then alternately spelled Kentucke County) was formed by the Commonwealth of Virginia by dividing Fincastle County into three new counties:
Kentucky, Washington, and Montgomery, effective December 31, 1776.

 During the three and one-half years of Kentucky County's existence, its seat of government was Harrodstown (then also known as Oldtown, later renamed
Harrodsburg).
</item 10>
=== END ITEM id=507c3cec9977033a ===

=== ITEM id=194a7da7aa48ce6c ===
Gold sentences and facts:
Sentence 0: Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that confirmed Trevi's status as a gay icon.
  Fact 0.0: Todos Me Miran ('Everyone looks at me') is a single from the Mexican artist Gloria Trevi.
  Fact 0.1: Todos Me Miran reached number 32 on Latin charts.
  Fact 0.2: Todos Me Miran became a club anthem.
  Fact 0.3: Todos Me Miran confirmed Trevi's status as a gay icon.
Sentence 1: Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
  Fact 1.0: Gloria Trevi was born Gloria de los Angeles Trevino Ruiz.
  Fact 1.1: Gloria Trevi was born on February 15, 1968.
  Fact 1.2: Gloria Trevi is a Mexican singer.
  Fact 1.3: Gloria Trevi is a Mexican songwriter.

Retrieved items:
<item 1>
Todos Los Romances
Todos Los Romances (English: "All the Romances" ) is a three-disc compilation album by Mexican singer Luis Miguel.

 Released on 11 August 1998 by WEA Latina, the record features the three previously released "Romance"-themed albums in which Miguel covered classic boleros in
each of them: "Romance" (1991), "Segundo Romance" (1994), and "Romances" (1997).

 An editor for AllMusic rated the album four of five stars.

 Commercially, "Todos Los Romances" peaked at number four in Spain and was certified double Platinum in the country.

 It also achieved Gold status in Argentina and peaked at number 12 on the "Billboard"'s Top Latin Albums in the United States.
</item 1>

<item 2>
De Contrabando
"De Contrabando" ("Smuggled") is a song written by Joan Sebastian.

 It was first performed by Mexican singer Imelda, and included on her album "Aparentemente", released in 1991.

 Six years later, was performed by fellow Mexican singer Zayda Y Los Culpables and released by Musart Records as the second single from her self-titled album in
1997.

 Mexican-American singer Jenni Rivera recorded a cover version for her studio album "Parrandera, Rebelde y Atrevida".

 "De Contrabando" was released as the second single from the album in 2006.

 In the United States, the song peaked at number fourteen on the "Billboard" Hot Latin Songs chart and number one on the "Billboard" Regional Mexican Songs
chart.

 The song was nominated for a Lo Nuestro Award for Regional Mexican Song of the Year at the 2007 Lo Nuestro Awards.

 Rivera was also recognized for Regional Mexican Female Artist of the Year.

 Rivera was also awarded Regional Mexican Airplay Song of the Year for at the 2007 Latin "Billboard" Music Awards.
</item 2>

<item 3>
Graph query row:
song: Todos Me Miran
person: Gloria Trevi
nationality: Mexican
occupation: singer
connection: CREATED_BY
connection_properties: {}
person_evidence: [{'chunk_index': 1031, 'title': 'Todos Me Miran'}, {'chunk_index': 1034, 'title': 'La Trayectoria (Gloria Trevi album)'}, {'chunk_index': 1738,
'title': 'Tu Ángel de la Guarda'}]
song_evidence: [{'chunk_index': 1031, 'title': 'Todos Me Miran'}, {'chunk_index': 1034, 'title': 'La Trayectoria (Gloria Trevi album)'}]
</item 3>

<item 4>
Mexico Espanol Airplay
Mexico Espanol Airplay is a record chart published weekly by "Billboard" magazine for Spanish language singles released in Mexico.

 According to "Billboard"' s electronic database, the first chart was published on October 1, 2011, with the track "Amor Clandestino" by Mexican band Maná at
number one; in the same year, fellow Mexican duo Jesse & Joy reached the top of the chart with their single "¡Corre!

", the first of their six number-one singles in the chart, the most for any act. "

¡Corre!"

 also won the Latin Grammy Awards for Record of the Year and Song of the Year in 2012.

 Starting on December 1, 2012, Mexican singer Thalía spent 10 non-consecutive weeks at number-one with "Manías", the first single from the album "Habítame
Siempre", which won the Lo Nuestro Award for Pop Album of the Year. "

Hoy Tengo Ganas de Ti" by Mexican artist Alejandro Fernández and American singer Christina Aguilera reached number-one in 2013, and also peaked at number four
in Spain.
</item 4>

<item 5>
Tu Ángel de la Guarda
Tu Ángel de la Guarda ("Your guardian angel") is Gloria Trevi's second album, and it contained one of her signature songs and her most widely known hit, "Pelo
Suelto".

 It also contained other hit songs such as "Tu angel de la guarda", "Ya no", "Virgen de las virgenes", and "Hoy me ire de casa".

 This album was very controversial different from other artists' albums such as Lucero.

 "Virgen de las virgenes" mocked girls who said they were virgins but had actually lost their virginity, and "¡Ya no!"

 went against the machismo movement of México.

 Following the release of the album, Gloria was working hard in promoting radio, television and print media.

 Trevi first traveled abroad visiting American Union countries, Puerto Rico, Argentina, Colombia, Guatemala, Venezuela, Costa Rica, and received a Gold and
Platinum for high sales achieved.
</item 5>

<item 6>
A Cambio de Que
"A Cambio de Que" (In Exchange for What) is a song written by Xavier Santos Córtes and first performed by Mexican singer Marisela from her studio album
"Completamente Tuya".

 The song has been covered by several artists which have ranked on the music charts.

 Latin pop band Corvo covered the song in 1999 which reached #18 on the Hot Latin Songs chart in the United States.

 Mexican band Palomo's cover peaked at #31 on the Regional Mexican Airplay chart.

 American singer Huey Dunbar recorded "A Cambio de Que" on his debut studio album "Yo Si Me Enamore" (2001).

 This version peaked at #16 on the Tropical Songs chart.

 Mexican duranguense band Alacranes Musical covered the song on the album of the same name in 2004 which reached #25 on the Regional Mexican Airplay chart.
</item 6>

<item 7>
Todos Me Miran
Todos Me Miran (""Everyone looks at me"") is a single from the Mexican artist Gloria Trevi reaching number 32 on Latin charts, and becoming a club anthem that
confirmed Trevi's status as a gay icon.

 The song, as interpreted in the music video, is about a young man who dares to crossdress in spite of society's opinions.
</item 7>

<item 8>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 8>

<item 9>
Gloria Trevi
Gloria Trevi (] ; born Gloria de los Ángeles Treviño Ruiz on February 15, 1968) is a Mexican singer and songwriter.
</item 9>

<item 10>
La Trayectoria (Gloria Trevi album)
La Trayectoria (In English: The Trajectory) is a 2006 double live CD and multimedia DVD release by the Mexican singer Gloria Trevi.

 The DVD includes scenes from parts of the tour from the Palacio de los Deportes of the Mexico City and The Arena Monterrey and the CD includes four new songs
like: "Todos Me Miran" (They All Look At Me) is the first single compilation study of Mexican singer "Gloria Trevi", this is an unreleased song from the album.
</item 10>
=== END ITEM id=194a7da7aa48ce6c ===

=== ITEM id=df68ac5675b63418 ===
Gold sentences and facts:
Sentence 0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.
  Fact 0.0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate.
  Fact 0.1: Suzanne N.J. 'Susie' Chun Oakland represents the 13th District.
  Fact 0.2: Suzanne N.J. 'Susie' Chun Oakland has represented the 13th District since 1996.
Sentence 1:  Previously she was a member of the Hawaii House of Representatives from 1990-96.
  Fact 1.0: Suzanne N.J. 'Susie' Chun Oakland was previously a member of the Hawaii House of Representatives.
  Fact 1.1: Suzanne N.J. 'Susie' Chun Oakland was a member of the Hawaii House of Representatives from 1990 to 1996.
Sentence 2:  The lower chamber of the legislature is the Hawaiʻi House of Representatives.
  Fact 2.0: The lower chamber of the Hawaii legislature is the Hawaiʻi House of Representatives.

Retrieved items:
<item 1>
Hawaii Senate
The Hawaiʻi State Senate is the upper chamber of the Hawaii State Legislature.

 The senate consists of twenty-five members elected from an equal number of constituent districts across the islands.

 The senate is led by the President of the Senate, elected from the membership of the body, currently Ron Kouchi.

 The forerunner of the Hawaii State Senate during the government of the Kingdom of Hawaiʻ i was the House of Nobles originated in 1840.

 In 1894 the Constitution of the Republic of Hawaii renamed the upper house the present senate.

 Senators are elected to four-year terms and are not subject to term limits.

 Like most state legislatures in the United States, the Hawaii State Senate is a part-time body and senators often have active careers outside government.

 The lower chamber of the legislature is the Hawaiʻi House of Representatives.

 The membership of the Senate also elects additional officers to include the Senate Vice President, Senate Chief Clerk, Assistant Chief Clerk, Senate Sergeant
at Arms and Assistant Sergeant at Arms.
</item 1>

<item 2>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 2>

<item 3>
Hawaii State Capitol
The Hawaii State Capitol is the official statehouse or capitol building of the U.S. state of Hawaii.

 From its chambers, the executive and legislative branches perform the duties involved in governing the state.

 The Hawaii State Legislature—composed of the twenty-five member Hawaii State Senate led by the President of the Senate and the fifty-one member Hawaii State
House of Representatives led by the Speaker of the House—convenes in the building.

 Its principal tenants are the Governor of Hawaii and Lieutenant Governor of Hawaii, as well as all legislative offices and the Legislative Reference Bureau.
</item 3>

<item 4>
Suzanne Chun Oakland
Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.

 Previously she was a member of the Hawaii House of Representatives from 1990-96.

 She is generally considered to have liberal political views.
</item 4>

<item 5>
Aloha Dalire
Fabianne Pomaialoha Wong Dalire, known professionally as Aloha Dalire, (June 22, 1950 – August 6, 2014) was an American Hawaiian kumu hula, or master hula
teacher.

 She won the first Miss Aloha Hula as Aloha Wong, in 1971, the same year that the Merrie Monarch Festival was established.

 The Miss Aloha Hula title is hula's top solo wahine (women's) honor.
</item 5>

<item 6>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 6>

<item 7>
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 7>

<item 8>
Alaska Legislature
The Alaska Legislature is the state legislature of the U.S. state of Alaska.

 It is a bicameral institution, consisting of the lower Alaska House of Representatives, with 40 members, and the upper house Alaska Senate, with 20 members.

 There are 40 House Districts (1-40) and 20 Senate Districts (A-T).

 With a total of 60 lawmakers, the Alaska Legislature is the smallest bicameral state legislature in the United States and the second-smallest of all state
legislatures (only the 49-member unicameral Nebraska Legislature is smaller).

 There are no term limits for either chamber.
</item 8>

<item 9>
Hawaii State Legislature
The Hawaii State Legislature is the state legislature of the U.S. state of Hawaii.

 The state legislature is a bicameral body consisting of a lower house, the Hawaii State House of Representatives, with 51 representatives, and an upper house,
the 25-member Hawaii State Senate.

 There are a total of 76 representatives in the legislature, each representing single member districts across the islands.

 The powers of the legislature are granted under Article III of the Constitution of Hawaii.
</item 9>

<item 10>
Oregon Territorial Legislature
Oregon’s Territorial Legislature was a bicameral legislative body created by the United States Congress in 1848 as the legislative branch of the government of
the Oregon Territory.

 The upper chamber Council and lower chamber House of Representatives first met in July 1849; they served as the region's legislative body until Oregon became a
state in February 1859, when they were replaced by the bicameral Oregon State Legislature.
</item 10>
=== END ITEM id=df68ac5675b63418 ===

=== ITEM id=3acee7a019e6115f ===
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
Pinhead Gunpowder (2008 EP)
Pinhead Gunpowder (also known by fans as the West Side Highway EP) is the second eponymously titled extended play by the American punk rock band Pinhead
Gunpowder.

 It was released on August 19, 2008, through Recess Records, the group's first release on Recess.

 "Pinhead Gunpowder" marked the band's first new release in over 5 years.

 The first pressings of the extended play were on green and blue vinyl.

 All of the tracks were rereleased on the 2009 compilation "Kick Over the Traces".
</item 1>

<item 2>
Green Day related projects
Since 1991, some members of punk band Green Day have branched out past their "main band" and have started other projects with other musicians and have released
full-length albums and several EPs.

 Notable related projects of Green Day include Billie Joe Armstrong's Pinhead Gunpowder (which also featured Green Day's other guitarist Jason White), The
Frustrators in which Mike Dirnt plays bass, and The Network which many speculate has all three members of Green Day, although under stagenames.
</item 2>

<item 3>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 3>

<item 4>
Compulsive Disclosure
Compulsive Disclosure is the second compilation album by the American punk rock band Pinhead Gunpowder.

 It was released on October 21, 2003, through Lookout!

 Records.

 The album features songs from the group's eponymously titled 2000 EP, the "Dillinger Four / Pinhead Gunpowder" split EP, the "8 Chords, 328 Words" EP and also
features re-recorded versions of the songs "2nd Street" and "At Your Funeral" (originally from "Dillinger Four / Pinhead Gunpowder").

 "Compulsive Disclosure" was re-released on CD and vinyl through Recess Records on February 12, 2010, with two unreleased tracks, "Salting Agents" and "El Lasso
Grappo".
</item 4>

<item 5>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 5>

<item 6>
Carry the Banner
Carry the Banner is the third EP by the Berkeley, California-based punk rock band Pinhead Gunpowder.

 Originally released on 10" vinyl in December 1994 through Too Many Records, the EP was reissued on CD by Lookout Records shortly after as the initial vinyl
pressing sold out quickly.

 It was the group's first release to feature Jason White on guitar/vocals, replacing Sarah Kirsch, who left the band in 1994 due to differences with Billie Joe
Armstrong after his main band Green Day signed to major label Reprise Records.
</item 6>

<item 7>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 7>

<item 8>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 8>

<item 9>
Bill Schneider (musician)
William 'Bill' Schneider (born June 17) is an American musician.

 He has played bass in the punk rock band Pinhead Gunpowder since 1990, as well as providing vocals and writing the lyrics to the song "Backyard Flames".

 He has also been in many other bands such as The Influents, Monsula and Uranium 9v, Sawhorse and The Skinflutes.

 Bill was worked as bass tech during the Nimrod era for Mike Dirnt, Green Day's and as guitar tech on their albums "Nimrod" and "Warning", American Idiot, 21st
Century Breakdown, Uno Dos Tres, Revolution Radio and was their day to day manager from 2003-2012 for their "American Idiot" and 21st Century Breakdown tours
through Uno Dos Tres.

 He also owned "Broken Guitars", a guitar shop in Oakland, California with Billie Joe Armstrong.

 Bill Still works on the sidelines with the band as well as in the studio.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=3acee7a019e6115f ===

=== ITEM id=62e7e071169757ae ===
Gold sentences and facts:
Sentence 0: Basie Meets Bond is a 1966 album by Count Basie and his orchestra.
  Fact 0.0: Basie Meets Bond is a 1966 album.
  Fact 0.1: Basie Meets Bond is an album by Count Basie.
  Fact 0.2: Basie Meets Bond is an album by Count Basie and his orchestra.
Sentence 1:  The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".
  Fact 1.0: The album is a collection of musical pieces.
  Fact 1.1: The album's musical pieces come from the first four James Bond films.
  Fact 1.2: The first four James Bond films include Dr No.
  Fact 1.3: The first four James Bond films include From Russia with Love.
  Fact 1.4: The first four James Bond films include Goldfinger.
  Fact 1.5: The first four James Bond films include Thunderball.
Sentence 2: William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.
  Fact 2.0: William James 'Count' Basie lived from August 21, 1904 to April 26, 1984.
  Fact 2.1: Count Basie was an American jazz pianist.
  Fact 2.2: Count Basie was an American organist.
  Fact 2.3: Count Basie was an American bandleader.
  Fact 2.4: Count Basie was an American composer.

Retrieved items:
<item 1>
The material surveys jazz musicians, styles, songs, and albums. Vince Guaraldi composed music for Peanuts and had a hit with “Cast Your Fate to the Wind.” Count
Basie’s career and recordings are covered, including “One O’Clock Jump,” The Atomic Mr. Basie, and Beatles- and Bond-themed albums. Other profiles include Miles
Davis, Don Ellis, Thad Jones, Louis Smith, blues and boogie-woogie pianists, and jazz fusion’s late-196
</item 1>

<item 2>
Basie's Beatle Bag
Basie's Beatle Bag is a 1966 studio album by Count Basie and his orchestra, arranged by Chico O'Farrill.

 Basie released a second album of Beatles songs, "Basie on the Beatles", in 1969.
</item 2>

<item 3>
Basie (album)
Basie (reissued as The Band of Distinction) is an album by pianist/bandleader Count Basie recorded in 1954 and originally released on the Clef label.

 the album should not be confused with Basie's 1958 album which became known as "The Atomic Mr. Basie".
</item 3>

<item 4>
Count Basie
William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.

 His mother taught him to play the piano and he started performing in his teens.

 Dropping out of school, he learned to operate lights for vaudeville and to improvise accompaniment for silent films at a local movie theater in his home town
of Red Bank, New Jersey.

 By age 16, he increasingly played jazz piano at parties, resorts and other venues.

 In 1924, he went to Harlem, where his performing career expanded; he toured with groups to the major jazz cities of Chicago, St. Louis and Kansas City.

 In 1929 he joined Bennie Moten's band in Kansas City, and played with them until Moten's death in 1935.
</item 4>

<item 5>
Basie on the Beatles
Basie on the Beatles is an album by pianist and bandleader Count Basie featuring performances recorded in late 1969 and released on the short-lived Happy Tiger
label.

 It was Basie's second album of Beatles' compositions following 1966's "Basie's Beatle Bag" and featured liner notes by Ringo Starr.
</item 5>

<item 6>
The Atomic Mr. Basie
The Atomic Mr. Basie (originally called Basie, also known as E=MC and reissued in 1994 as The Complete Atomic Basie) is a 1958 album by Count Basie and his
orchestra.

 The album is one of Basie's most famous and is critically acclaimed.

 Allmusic gave it 5 stars, reviewer Bruce Eder saying "it took Basie's core audience and a lot of other people by surprise, as a bold, forward-looking statement
within the context of a big-band recording."

 It is included in the book "1001 Albums You Must Hear Before You Die", Will Fulford-Jones calling it "Basie's last great record."

 According to the 2015 site update of Acclaimed Music, it is the 6th most critically acclaimed album of 1958, the 25th most acclaimed of the 1950s, and the
837th most acclaimed of all time, based on an aggregation of hundreds of critics' lists from around the world.
</item 6>

<item 7>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 7>

<item 8>
Basie Meets Bond
Basie Meets Bond is a 1966 album by Count Basie and his orchestra.

 The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".

 The album was Basie's first for United Records, and was produced by Teddy Reig.
</item 8>

<item 9>
Count Basie at Newport
Count Basie at Newport is a live album by jazz musician Count Basie and his orchestra.

 It was originally issued as Verve MGV 8243 and included only the tracks 1-8 and 13.

 Tracks 9-12 originally included in "Count Basie & Joe Williams/Dizzy Gillespie & Mary Lou Williams at Newport" (Verve MGV 8244).
</item 9>

<item 10>
Count Basie/Sarah Vaughan
Count Basie/Sarah Vaughan is a 1961 album by American jazz singer Sarah Vaughan, accompanied by the Count Basie Orchestra, with arrangements by Frank Foster,
Thad Jones and Ernie Wilkins.

 According to James Gavin's liner notes to the 1996 CD release, Basie himself does not perform on any of the tracks.
</item 10>
=== END ITEM id=62e7e071169757ae ===

=== ITEM id=5058c99c5e2f7540 ===
Gold sentences and facts:
Sentence 0: Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from 1997-1998.
  Fact 0.0: Robert Suettinger was the national intelligence officer for East Asia at the National Intelligence Council (NIC).
  Fact 0.1: Robert Suettinger held the role of national intelligence officer for East Asia for United States President Bill Clinton.
  Fact 0.2: Robert Suettinger held the role from 1997 to 1998.
Sentence 1: William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States from 1993 to 2001.
  Fact 1.0: William Jefferson Clinton was born William Jefferson Blythe III.
  Fact 1.1: William Jefferson Clinton was born on August 19, 1946.
  Fact 1.2: William Jefferson Clinton is an American politician.
  Fact 1.3: William Jefferson Clinton served as the 42nd President of the United States.
  Fact 1.4: William Jefferson Clinton served as President from 1993 to 2001.
Sentence 2:  Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.
  Fact 2.0: Before the presidency, William Jefferson Clinton was Governor of Arkansas.
  Fact 2.1: William Jefferson Clinton was Governor of Arkansas during two separate terms.
  Fact 2.2: William Jefferson Clinton's first term as Governor of Arkansas was from 1979 to 1981.
  Fact 2.3: William Jefferson Clinton's second term as Governor of Arkansas was from 1983 to 1992.

Retrieved items:
<item 1>
Steven K. Galson
Steven Kenneth Galson (born 1956) is an American public health physician.

 He is currently Senior Vice President for Global Regulatory Affairs at Amgen, the S. California-based innovative global biopharmaceutical company.

 He is also Professor-at-Large at the Keck Graduate Institute for Applied Life Sciences in Clarmont, California.

 He is a retired rear admiral in the United States Public Health Service Commissioned Corps and public health administrator who served as the acting Surgeon
General of the United States from October 1, 2007 – October 1, 2009.

 He served concurrently as acting Assistant Secretary for Health from January 22, 2009 to June 25, 2009, and as the Deputy Director and Director of the Center
for Drug Evaluation and Research (CDER) at the Food and Drug Administration from 2001 to 2007.

 As the Acting Surgeon General, he was the commander of the United States Public Health Service Commissioned Corps and, while serving as the Assistant Secretary
for Health, was the operational head of the Public Health Service.
</item 1>

<item 2>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 2>

<item 3>
Susan Combs
Susan Combs (born February 26, 1945) is a Republican politician from the U.S. state of Texas, who served from 2007 to 2015 as the state's Comptroller of Public
Accounts.

 Prior to her tenure as Comptroller, Combs had served two terms as Commissioner of the Texas Department of Agriculture from 1999–2007, taking the reins as the
first woman elected to that office in 1998.

 Combs also served two terms in the Texas House of Representatives.

 On July 10, 2017, Combs was nominated by U.S. President Donald Trump to be the Assistant Secretary of the Interior for Policy, Management and Budget.
</item 3>

<item 4>
Bill Clinton
William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States
from 1993 to 2001.

 Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.

 Before that, he served as Arkansas attorney general, from 1977 to 1979.

 A member of the Democratic Party, Clinton was ideologically a New Democrat and many of his policies reflected a centrist "Third Way" political philosophy.
</item 4>

<item 5>
Lawrence Eagleburger
Lawrence Sidney Eagleburger (August 1, 1930 – June 4, 2011) was an American statesman and career diplomat, who served briefly as the Secretary of State under
President George H. W. Bush.

 Previously, he had served in lesser capacities under Presidents Richard Nixon, Jimmy Carter, and Ronald Reagan, and as Deputy Secretary of State under George
H. W. Bush.
</item 5>

<item 6>
Director of the Defense Intelligence Agency
The Director of the Defense Intelligence Agency is an intelligence officer who, upon nomination by the President and confirmation by the Senate, serves as the
nation's highest-ranking military intelligence officer.

 As the chief of the Defense Intelligence Agency, the Director is the primary intelligence adviser to the Secretary of Defense and the Chairman of the Joint
Chiefs of Staff and also answers to the Director of National Intelligence through the civilian Under Secretary of Defense for Intelligence.

 The Director is also the Commander of the Joint Functional Component Command for Intelligence, Surveillance and Reconnaissance, a subordinate command of United
States Strategic Command.

 Additionally, he chairs the Military Intelligence Board, which coordinates activities of the entire defense intelligence community.
</item 6>

<item 7>
Wayne W. Lambert
The Brigadier General retired from the US Air Force in 1989.

 Lambert also holds an MBA from Chapman College (’76).

 In retirement, Lambert publicly supported U.S. presidential candidate Mitt Romney and U.S. vice presidential candidate Paul Ryan in the 2012 general election.
</item 7>

<item 8>
George G. Finch
Army in September, 1941, with Major Finch as commander.

 After World War II, he was a leading critic of efforts to eliminate the air arm of the National Guard during peacetime.

 General Finch gained the respect and admiration of Air National Guardsmen throughout the nation with his steadfast support and successful efforts to preserve
the Air Guard.

 He became the first Chief of the Air Force Division of the National Guard Bureau in 1948.

 Under his leadership, the Air National Guard built to combat readiness and was among the first components called into service after the outbreak of the Korean
War.

 As a result of General Finch's vision and perseverance, 45,000 highly trained officers and airmen of 22 wings and 65 squadrons gave the Air Force the strength
it needed in the early, critical phases of the Communist drive down the Korean peninsula.General Finch served as the senior Air Force member of the United
Nations negotiating team at the peace talks at Panmunjom, Korea, and received the Legion of Merit for outstanding service in 1955; General Finch assumed command
of Fourteenth Air Force, Robins AFB, Georgia, becoming the nation's first Air National Guardsman to head a numbered air force.
</item 8>

<item 9>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 9>

<item 10>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 10>
=== END ITEM id=5058c99c5e2f7540 ===

=== ITEM id=e6e126669b36af84 ===
Gold sentences and facts:
Sentence 0: Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.
  Fact 0.0: Cantarella was a poison
  Fact 0.1: Cantarella was allegedly used by the Borgias
  Fact 0.2: The alleged use of Cantarella by the Borgias occurred during the papacy of Pope Alexander VI
Sentence 1:  The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to remove his name from the records.
  Fact 1.0: The use of Cantarella is not well documented in any of the papal records
  Fact 1.1: Cantarella was most likely conceived after 1503
  Fact 1.2: Cantarella was most likely conceived as part of Pope Julius II's effort to remove his name from the records
Sentence 2: Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
  Fact 2.0: Damnatio memoriae is a modern Latin phrase
  Fact 2.1: Damnatio memoriae literally means condemnation of memory
  Fact 2.2: Damnatio memoriae means that a person must not be remembered

Retrieved items:
<item 1>
Pedro Luis de Borja Lanzol de Romaní
Pedro Luis de Borja Lanzol de Romaní, O.S.Io.Hieros.

 (1472–1511) was a Roman Catholic cardinal and cardinal-nephew and papal military leader.

 He received a wide variety of sinecures during the papacy of his great-uncle, Pope Alexander VI, but was exiled to Naples on the election of Borja rival Pope
Julius II.

 Borja also fought with the Knights Hospitaller in Jerusalem and Rhodes.
</item 1>

<item 2>
Damnatio memoriae
Damnatio memoriae is a modern Latin phrase literally meaning "condemnation of memory", meaning that a person must not be remembered.
</item 2>

<item 3>
Dictatus papae
Dictatus papae is a compilation of 27 statements of powers arrogated to the Pope that was included in Pope Gregory VII's register under the year 1075.

 Some historians argue that it was written (or dictated) by Gregory VII himself; others argue that it has been inserted in the register at a later date, and
that it had a different origin.

 In 1087 Cardinal Deusdedit published a collection of decretals, dedicated to Pope Victor III, that embodied the law of the Church – Canon law – which he had
compiled from many sources, both legitimate and false (see Pseudo-Isidore).

 The "Dictatus papae" agrees so clearly and closely with this collection that some have argued the "Dictatus" must have been based on it; and so must be of a
later date of compilation and insertion in the papal register than 1087.
</item 3>

<item 4>
Sicut Dudum
Sicut Dudum (English: "Just As Long Ago" ) is a papal bull promulgated by Pope Eugene IV in Florence on January 13, 1435, which forbade the enslavement of local
natives in the Canary Islands who had converted or were converting to Christianity.

 "Sicut Dudum" was meant to reinforce "Creator Omnium", issued the previous year, condemning Portuguese slave raids in the Canary Islands.

 Over forty years after "Creator Omnium" and "Sicut Dudum, Pope Sixtus IV found it necessary repeat the prohibition in his papal bull "Regimini Gregis" which
threatened the excommunication of all captains or pirates who enslaved Christians.
</item 4>

<item 5>
Pope Julius II
Pope Julius II (Italian: "Papa Giulio II" ; Latin: "Iulius II" ) (5 December 1443 – 21 February 1513), born Giuliano della Rovere, and nicknamed "The Fearsome
Pope" and "The Warrior Pope", was Pope from 1 November 1503 to his death in 1513.

 His papacy was marked by an active foreign policy, ambitious building projects, and patronage of the arts—he commissioned the destruction and rebuilding of St.
Peter's Basilica, and Michelangelo's decoration of the ceiling of the Sistine Chapel.

 In addition to an active military policy, he personally led troops into battle on at least two occasions, the first to expel Giovanni Bentivoglio from Bologna
(17 August 1506–23 March 1507), and the second in an attempt to recover Ferrara for the Papal States (1 September 1510–29 June 1512).
</item 5>

<item 6>
Cantarella
Cantarella was a poison allegedly used by the Borgias during the papacy of Pope Alexander VI.

 It was probably a variation of arsenic or cantharidin powder (made from blister beetles) .

 The use of this poison is not well documented in any of the papal records and it was most likely conceived after 1503 as part of Pope Julius II's effort to
remove his name from the records.
</item 6>

<item 7>
Giovanni Borgia (Infans Romanus)
Giovanni Borgia (March 1498 – 1548), known as the Infans Romanus ("the Roman child"), was born into the House of Borgia in secret and is of unclear parentage.

 Speculations of the child's parentage involve either Lucrezia Borgia with her alleged lover, Perotto Calderon or Cesare Borgia, or Pope Alexander VI as his
father.

 Cesare Borgia's biographer Rafael Sabatini says that the truth is fairly clear: Alexander fathered the child with an unknown Roman woman.
</item 7>

<item 8>
Papal conclave, September 1503
The papal conclave of September 1503 elected Pope Pius III to succeed Pope Alexander VI.

 Due to the Italian Wars, the College of Cardinals was surrounded by three potentially hostile armies, loyal to Louis XII of France, Ferdinand II of Aragon, and
Cesare Borgia (the cardinal-nephew of Alexander VI).
</item 8>

<item 9>
Portrait of Pope Julius II
Portrait of Pope Julius II is an oil painting of 1511–12 by the Italian High Renaissance painter Raphael.

 The portrait of Pope Julius II was unusual for its time and would carry a long influence on papal portraiture.

 From early in its life, it was specially hung at the pillars of the church of Santa Maria del Popolo, on the main route from the north into Rome, on feast and
high holy days.

 Giorgio Vasari, writing long after Julius' death, said that "it was so lifelike and true it frightened everyone who saw it, as if it were the living man
himself".
</item 9>

<item 10>
Dum Diversas
Dum Diversas (English: "Until different") is a papal bull issued on 18 June 1452 by Pope Nicholas V.

 It authorized Afonso V of Portugal to conquer Saracens and pagans and consign them to "perpetual servitude".

 Pope Calixtus III reiterated the bull in 1456 with "Inter Caetera" (not to be confused with Alexander VI's), renewed by Pope Sixtus IV in 1481 and Pope Leo X
in 1514 with "Precelse denotionis".

 The concept of the consignment of exclusive spheres of influence to certain nation states was extended to the Americas in 1493 by Pope Alexander VI with "Inter
caetera".
</item 10>
=== END ITEM id=e6e126669b36af84 ===
