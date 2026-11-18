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

=== ITEM id=c324dddb5e633bb9 ===
Gold sentences and facts:
Sentence 0: Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
  Fact 0.0: Milium's vernacular name is milletgrass.
  Fact 0.1: Milium is a genus of plants in the grass family.
  Fact 0.2: Milium plants are Eurasian, North American, and North African.
Sentence 1: Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.
  Fact 1.0: Eucommia is a genus of small trees.
  Fact 1.1: Eucommia trees are now native to China.
  Fact 1.2: The fossil record of Eucommia shows a much wider distribution.
Sentence 2:  The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in herbology such as traditional Chinese medicine.
  Fact 2.0: Eucommia has a single living species, Eucommia ulmoides.
  Fact 2.1: Eucommia ulmoides is near threatened in the wild.
  Fact 2.2: Eucommia ulmoides is widely cultivated in China for its bark.
  Fact 2.3: Eucommia ulmoides is highly valued in herbology, such as traditional Chinese medicine.

Retrieved items:
<item 1>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 1>

<item 2>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 2>

<item 3>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 3>

<item 4>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 4>

<item 5>
Graph query row:
entity: Eucommia
relationship: LOCATED_IN
related: China
</item 5>

<item 6>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 6>

<item 7>
Ocimum
Ocimum is a genus of aromatic annual and perennial herbs and shrubs in the family Lamiaceae, native to the tropical and warm temperate regions of all 6
inhabited continents, with the greatest number of species in Africa.

 Its best known species are the cooking herb Cooking basil, "O. basilicum" and the medicinal herb Tulsi (holy basil), "O. tenuiflorum".
</item 7>

<item 8>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 8>

<item 9>
Ardisia
Ardisia (coralberry or marlberry) is a genus of flowering plants in the family Primulaceae.

 It was in the former Myrsinaceae family now recognised as the myrsine sub-family Myrsinoideae.

 They are distributed in the Americas, Asia, Australia, and the Pacific Islands, mainly in the tropics.

 There are 400 to 500 species.
</item 9>

<item 10>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 10>
=== END ITEM id=c324dddb5e633bb9 ===

=== ITEM id=69aa8700b3d1a165 ===
Gold sentences and facts:
Sentence 0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.
  Fact 0.0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music.
  Fact 0.1: More Hot Rocks was released by ABKCO Records.
  Fact 0.2: ABKCO Records is run by former manager Allen Klein.
  Fact 0.3: Allen Klein usurped control of the Rolling Stones' Decca/London material in 1970.
  Fact 0.4: More Hot Rocks was released after the Rolling Stones departed from Decca.
  Fact 0.5: More Hot Rocks was released after the Rolling Stones departed from Allen Klein.
Sentence 1: Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording artists.
  Fact 1.0: Allen Klein was born December 18, 1931.
  Fact 1.1: Allen Klein died July 4, 2009.
  Fact 1.2: Allen Klein was an American businessman.
  Fact 1.3: Allen Klein was a music publisher.
  Fact 1.4: Allen Klein was a writers' representative.
  Fact 1.5: Allen Klein was a filmmaker.
  Fact 1.6: Allen Klein was a record label executive.
  Fact 1.7: Allen Klein was most noted for his tough persona.
  Fact 1.8: Allen Klein was most noted for aggressive, innovative negotiation tactics.
  Fact 1.9: Many of Allen Klein's negotiation tactics established higher industry standards for compensating recording artists.
Sentence 2:  He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming one of the most powerful individuals in the music industry during his era.
  Fact 2.0: Allen Klein first scored massive monetary windfalls for Buddy Knox.
  Fact 2.1: Allen Klein first scored massive contractual windfalls for Buddy Knox.
  Fact 2.2: Allen Klein first scored massive monetary windfalls for Jimmy Bowen.
  Fact 2.3: Allen Klein first scored massive contractual windfalls for Jimmy Bowen.
  Fact 2.4: Buddy Knox and Jimmy Bowen were one-hit rockabillies of the late 1950s.
  Fact 2.5: Allen Klein later parlayed his early successes into a position managing Sam Cooke.
  Fact 2.6: Allen Klein eventually managed the Beatles.
  Fact 2.7: Allen Klein eventually managed the Rolling Stones.
  Fact 2.8: Allen Klein managed the Beatles and the Rolling Stones simultaneously.
  Fact 2.9: Allen Klein managed many other artists.
  Fact 2.10: Allen Klein became one of the most powerful individuals in the music industry during his era.

Retrieved items:
<item 1>
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 1>

<item 2>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 2>

<item 3>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 3>

<item 4>
More Hot Rocks (Big Hits &amp; Fazed Cookies)
More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who
usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1972 as the follow-up to the hugely successful "Hot Rocks 1964–1971", it was another success for Klein and ABKCO.
</item 4>

<item 5>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 5>

<item 6>
Hot Rocks 1964–1971
Hot Rocks 1964–1971 is the first compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who gained control of the
band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1971, it proved to be The Rolling Stones' biggest-selling release of their career and an enduring and popular retrospective.
</item 6>

<item 7>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 7>

<item 8>
A Compilation of Scott Weiland Cover Songs
A Compilation of Scott Weiland Cover Songs is a compilation album and third solo studio album by Stone Temple Pilots frontman Scott Weiland, released on August
30, 2011.

 The album consists entirely of cover songs, featuring an array of covers of artists that inspired Weiland, such as David Bowie, The Beatles, The Rolling
Stones, and The Smiths.

 The album was originally set to be released together with Weiland's memoir "Not Dead & Not for Sale" but Weiland decided later to release the album by itself
exclusively in digital format.
</item 8>

<item 9>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 9>

<item 10>
Lonesome Pine Fiddlers
None of them ever had a music lesson yet excelled on every instrument they touched.

 Natives of the Gilbert Creek region of southern West Virginia, Cousin Ezra, along with brothers Ireland (Lazy Ned) and Curly Ray Cline, were part of the
original Lonesome Pine Fiddlers from about 1938, a group that worked on radio at WHIS Bluefield, West Virginia.

 During World War II, Ned was killed in action.

 When the Pine Fiddlers resumed regular daily broadcasts, Charlie, who played multiple instruments, joined them on a regular basis.

 Charlie returned to the Fiddlers briefly before becoming a member of Bill Monroe's Blue Grass Boys.

 During 1952-1955, Charlie worked off and on with Monroe, recording some 38 songs, all on Decca.

 It has been said that he played every instrument at one time or another in the Monroe group except mandolin.

 Charlie spent most of 1953 back with the Lonesome Pine Fiddlers working at WJR radio in Detroit.

 When Ezra brought the band to Pikeville, Kentucky, in November, Charlie rejoined Bill Monroe.
</item 10>
=== END ITEM id=69aa8700b3d1a165 ===

=== ITEM id=a5d37d9eab0aed97 ===
Gold sentences and facts:
Sentence 0: Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
  Fact 0.0: Distortion Mirrors is an English alternative rock band.
  Fact 0.1: Distortion Mirrors is based in Leeds, West Yorkshire, England.
  Fact 0.2: Distortion Mirrors was formed in 2011.
Sentence 1: OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.
  Fact 1.0: OK Go is an American rock band.
  Fact 1.1: OK Go was originally from Chicago, Illinois.
  Fact 1.2: OK Go is now based in Los Angeles, California.

Retrieved items:
<item 1>
Twelve Days of OK Go
Twelve Days of OK Go is a compilation album by American rock band OK Go.

 It was released on December 31, 2012.

 OK Go started releasing the songs on December 10, with one song released each weekday.

 The last song, a cover of "Any Time at All", was released on Christmas.

 A bonus track, a cover of "This Will Be Our Year," was released on New Year's Eve.
</item 1>

<item 2>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 2>

<item 3>
OK Go videography
The musical rock band OK Go has earned considerable fame for their creative but often low-budget music videos, most of which have been promoted through Internet
video sharing sites like YouTube.

 Many of these have become viral videos; the 2006 video for "Here It Goes Again", in which the band performed a complex routine with the aid of motorized
treadmills, has received over 50 million views four years later.

 Their video for Needing/Getting, released February 5, 2012 in partnership with Chevrolet, debuted during Super Bowl XLVI and has over 32 million views on
YouTube.

 Samuel Bayer, who produced many music videos in the 1990s, asserted that OK Go's promotion of music videos on the Internet was akin to Nirvana's ushering in
the grunge movement.

 Many of the videos also use long or single-shot takes, which "Salon"' s Matt Zoller Seitz says "restore[s] a sense of wonder to the musical number by letting
the performers' humanity shine through and allowing them to do their thing with a minimum of filmmaking interference".
</item 3>

<item 4>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 4>

<item 5>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 5>

<item 6>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 6>

<item 7>
The Writing's on the Wall (OK Go song)
"The Writing's on the Wall" is a song by American rock band OK Go.

 It was released on June 17, 2014, as part of the band's EP "Upside Out", and is also the first single from the band's fourth studio album "Hungry Ghosts".

 On the same day, the band released a music video in which the members use props to create optical illusions, reflecting the song's description of a
relationship that fails because the couple has different points of view.

 Like previous OK Go videos, it is structured as a one-shot music video.

 The many YouTube views of the video caused the song to debut in the top ten of the US "Billboard" Hot Rock Songs chart, as well as number one on the Bubbling
Under Hot 100 Singles chart.
</item 7>

<item 8>
OK Go videography
The success of OK Go's music first won the band the 14th Annual Webby Special Achievement Award for Film and Video Artist of the Year.

 The video for "This Too Shall Pass" was named both "Video of the Year" and "Best Rock Video" at the 3rd annual UK Music Video Awards.

"This Too Shall Pass" won the LA Film Fest's Audience Award for Best Music Video, UK MVA Awards – Music Video of the Year Winner 2010, among others.
</item 8>

<item 9>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 9>

<item 10>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 10>
=== END ITEM id=a5d37d9eab0aed97 ===

=== ITEM id=b12162f1e7068522 ===
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
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 1>

<item 2>
George McMillin
George Johnson McMillin (November 25, 1889 – August 29, 1983 ) was a United States Navy Rear Admiral who served as the 38th and final Naval Governor of Guam.

 He graduated from the United States Naval Academy in 1911 and served as an officer during four separate conflicts: World War I, the occupation of the Dominican
Republic, the United States occupation of Veracruz, and World War II.

 He served on the staff of both the Naval Academy and the Naval War College as well.

 He is most remembered as the commander who surrendered Guamanian forces to a much larger Japanese force during the First Battle of Guam, only the second battle
of World War II involving the United States.

 He had previously evacuated all but one civilian American citizen from the island and attempted to rebuild defenses after a strong typhoon devastated the
island the year before.

 On December 8, 1941, Japanese forces invaded Guam and McMillin surrendered two days later.

 He spent the rest of the war at various Japanese prisoner of war camps.
</item 2>

<item 3>
Julia Compton Moore
Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)

 Hal Moore, a United States Army officer.

 Her efforts and complaints in the aftermath of the Battle of Ia Drang prompted the U.S. Army to set up survivor support networks and casualty notification
teams consisting of uniformed officers, which are still in use.
</item 3>

<item 4>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 4>

<item 5>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 5>

<item 6>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 6>

<item 7>
Mark E. Mitchell
Mark E. Mitchell is the current Acting Assistant Secretary of Defense for Special Operations and Low-Intensity Conflict of the United States Department of
Defense.

 A retired Colonel, Mitchell was the first member of the United States Army to be awarded the Distinguished Service Cross during the War in Afghanistan and was
the first to receive the award since the Vietnam War.

 In 2003, he received the award for his actions during the Battle of Qala-i-Jangi, which took place in late November – early December 2001.
</item 7>

<item 8>
William K. MacNulty
William Kirk MacNulty (May 22, 1892 - August 3, 1964) was a U. S. Marine with a long and distinguished record.

 He was a young U.S. Marine Corps Second Lieutenant during World War One and saw action at the Battle of the Argonne Forest.

 He served as a captain during the Second U.S. Nicaraguan Campaign (1926–33).

 During the Second World War as Lieutenant Colonel he commanded the U.S. Marine Corps defense of Guam against overwhelming Imperial Japanese forces during the
First Battle of Guam.

 He was incarcerated by the Japanese as a prisoners of war.

 He was promoted to Brigadier General during captivity and retired from military service in 1946.

 He is buried at the Golden Gate National Cemetery in San Bruno, San Mateo County, California.
</item 8>

<item 9>
Myron F. Diduryk
Myron F. Diduryk (July 15, 1938 to April 24, 1970) was a Ukrainian-American United States Army Major, who played a key role as an infantry company commander in
the Battle of Ia Drang, the first major battle of the Vietnam War.

 His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".

 Moore said that Diduryk was, “… the finest battlefield company commander I had ever seen, bar none.”

 Diduryk was killed in action on his second tour in Vietnam.
</item 9>

<item 10>
Matthew Ridgway
General Matthew Bunker Ridgway (March 3, 1895 – July 26, 1993) was the 19 Chief of Staff of the United States Army.

 He served with great distinction during World War II, where he was the Commanding General (CG) of the 82nd Airborne Division, leading it in action in Sicily,
Italy and Normandy, before taking command of the newly formed XVIII Airborne Corps in August 1944, holding this post until the end of the war, commanding it in
the Battle of the Bulge, Operation Varsity and the Western Allied invasion of Germany.
</item 10>
=== END ITEM id=b12162f1e7068522 ===

=== ITEM id=e954009368575624 ===
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
Myles Kennedy
Myles Richard Kennedy (born Myles Richard Bass; November 27, 1969) is an American musician, singer, and songwriter, best known as the lead vocalist and rhythm
guitarist of the rock band Alter Bridge, and as the lead vocalist in guitarist Slash's backing band, known as Myles Kennedy and the Conspirators.

 A former guitar instructor from Spokane, Washington, he has worked as a session musician and songwriter, making both studio and live appearances with several
artists, and has been involved with several projects throughout his career.
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
</item 4>

<item 5>
Cosmic Dust (band)
Cosmic Dust, also known as the Cosmic Dust Fusion Band, is an instrumental jazz band formed in 1990 by Jim Templeton.

 The band was the first well-known group that guitarist Myles Kennedy played in.

 The original lineup consisted of Jim Templeton on keyboard, Gary Edighoffer on saxophone, Clipper Anderson on double bass, Myles Kennedy on guitar, and Scott
Reusser on drums.

 Kennedy eventually left the band and went on to become the lead vocalist/lead guitarist for a jazz fusion group called Citizen Swing and later an alternative
rock band called The Mayfield Four.

 Kennedy is now fronting and playing guitar for the hard rock/alternative metal band Alter Bridge, which he helped form with Creed members Mark Tremonti, Scott
Phillips, and Brian Marshall in 2004, and is also the lead vocalist for Slash's solo band on tour.
</item 5>

<item 6>
List of songs recorded by Myles Kennedy
Myles Kennedy is an American rock musician.

 Born in Boston, Massachusetts, he originally began his musical career in Spokane, Washington as the guitarist in jazz band Cosmic Dust.

 After recording 1991's "Journey", he left to form alternative rock band Citizen Swing in 1992.

 The group released two albums, 1992's "Cure Me with the Groove" and 1996's "Deep Down", on both of which Kennedy was co-credited for songwriting.

 Kennedy's next band was The Mayfield Four, which he co-founded with guitarist Craig Johnson, bassist Marty Meisner and drummer Zia Uddin in 1996.

 The frontman wrote most of the material on the band's 1998 debut "Fallout" and all of the songs on 2001's "Second Skin".
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
relationship: BORN_IN
connected_role: None
related_entity: Boston
</item 9>

<item 10>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 10>
=== END ITEM id=e954009368575624 ===

=== ITEM id=50cf2c2f10317d88 ===
Gold sentences and facts:
Sentence 0:  Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great America).
  Fact 0.0: The roller coaster was built in 1988 by the TOGO company.
  Fact 0.1: The roller coaster built by the TOGO company was the fastest roller coaster in the world when it was built.
  Fact 0.2: The roller coaster built by the TOGO company took the record for fastest roller coaster in the world from American Eagle at Six Flags Great America.
Sentence 1: TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky cycles and other amusement rides.
  Fact 1.0: TOGO (株式会社トーゴ, Kabushiki-gaisha Tōgo) was a Japanese amusement ride company.
  Fact 1.1: TOGO built roller coasters.
  Fact 1.2: TOGO built giant wheels.
  Fact 1.3: TOGO built carousels.
  Fact 1.4: TOGO built flumes.
  Fact 1.5: TOGO built dark rides.
  Fact 1.6: TOGO built sky cycles.
  Fact 1.7: TOGO built other amusement rides.

Retrieved items:
<item 1>
White Cyclone
White Cyclone (ホワイトサイクロン , Howaito Saikuron ) is a wooden roller coaster at Nagashima Spa Land in Mie Prefecture, Japan.

 At 1700 m in length, White Cyclone is the third longest wooden roller coaster in the world, and is the longest wooden roller coaster outside of the United
States.

 Despite its length, White Cyclone is still considerably shorter than the 2479 m Steel Dragon 2000, the world's longest steel roller coaster, which is also at
Nagashima Spa Land.

 In addition to being the third longest wooden roller coaster, White Cyclone is the seventh tallest wooden roller coaster in the world and the fourth tallest
wooden roller coaster outside the United States.

 A single ride on the White Cyclone costs ¥1,000 (approximately $9 USD), and the ride is restricted to those individuals above 1.3 m in height; and those
individuals under 54 years of age.
</item 1>

<item 2>
Giovanola
Giovanola Freres SA was one of the prominent thrill ride manufacturers in the world.

 It was well known for thrill rides and also built electrical power stations, water storage tanks, pipelines, highway bridges, and many other steel products.

 The company started out as a small metal forging shop, founded by Joseph Giovanola in 1888.

 It served as a subcontractor to Intamin supplying rides and roller coasters.

 In 1998 Giovanola started marketing directly under the name Giovanola Amusement Rides Worldwide.

 The company was based in Monthey, Switzerland.

 During its last years in business, Giovanola fashioned steel behind the scenes for companies such as Intamin and Bolliger & Mabillard.

 The company also built its own thrill rides and roller coasters from 1998 to 2001.

 Examples of Giovanola roller coasters are Goliath at Six Flags Magic Mountain, Titan at Six Flags Over Texas and Anaconda at South Africa's Gold Reef City.
</item 2>

<item 3>
Pretzel Amusement Ride Company
The Pretzel Amusement Ride Company was a famous manufacturer of pretzel dark rides. Pretzel built over 1400 pretzel rides and sold them to carnivals and parks.

 The pretzel ride was invented by Marvin Rempfer.

 Leon Cassidy was Marvin's partner in the Pretzel Amusement Ride Company.

 Both names are on the patent.

 Leon Cassidy patented the single-rail dark ride in 1928 along with Marvin Rempfer.

 The company originated in Tumbling Dam Park on the banks of Sunset Lake in Bridgeton, New Jersey.

 A rider said that ""It felt like I was turned and twisted like a Pretzel"", so the name Pretzel was chosen.

 A large heavy pretzel design was originally affixed to the front of each car to prevent the car from flipping backwards.

 In 1929, a standard Pretzel ride had five cars, 350 feet of track, and was one and a half minutes per ride.

 A pretzel ride sold for $1,200.
</item 3>

<item 4>
Mack Rides
Mack Rides GmbH & Co KG, also known simply as Mack Rides, is a German company that designs and constructs amusement rides.

 Mack Rides in headquartered in Waldkirch, Baden-Württemberg, Germany.

 Mack Rides is one of the world's oldest amusement industry suppliers and builds all kinds of amusement devices including several types of flat rides, dark
rides, log flumes, tow boat rides and roller coasters.

 The family that owns Mack Rides also owns Europa-Park.
</item 4>

<item 5>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 5>

<item 6>
List of roller coaster rankings
Roller coasters are amusement rides developed for amusement parks and modern theme parks.

 During the 16th and 17th centuries, rides consisting of wooden sleds that took riders down large slides made from ice were popular in Russia.

 The first roller coasters, where the train was attached to a wooden track, first appeared in France in the early 1800s.

 Although wooden roller coasters are still being produced, steel roller coasters are more common and can be found on every continent except Antarctica.
</item 6>

<item 7>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 7>

<item 8>
Bandit (Yomiuriland)
Bandit (バンデット , Bandetto ) is a steel roller coaster located at Yomiuriland in the city of Inagi, near Tokyo, Japan.

 Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great
America).

 It lost the record to Magnum XL-200 at Cedar Point one year later.
</item 8>

<item 9>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 9>

<item 10>
TOGO
TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky
cycles and other amusement rides.
</item 10>
=== END ITEM id=50cf2c2f10317d88 ===

=== ITEM id=833cb3597da127b2 ===
Gold sentences and facts:
Sentence 0: Bank Street () is a short street in Central, Hong Kong.
  Fact 0.0: Bank Street is a short street.
  Fact 0.1: Bank Street is located in Central, Hong Kong.
Sentence 1:  It links Des Voeux Road Central to Queen's Road Central.
  Fact 1.0: Bank Street links Des Voeux Road Central to Queen's Road Central.
Sentence 2:  It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong) respectively.
  Fact 2.0: Bank Street is named after its location between the HSBC Building and the Bank of China Building.
  Fact 2.1: The HSBC Building was the former headquarters of HSBC.
  Fact 2.2: The Bank of China Building was the former headquarters of Bank of China (Hong Kong).
Sentence 3: HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC Holdings.
  Fact 3.0: HSBC Main Building is a headquarters building of The Hongkong and Shanghai Banking Corporation.
  Fact 3.1: The Hongkong and Shanghai Banking Corporation is today a wholly owned subsidiary of London-based HSBC Holdings.

Retrieved items:
<item 1>
Bank of Canton
Bank of Canton () was established in 1912 in Canton but registered as a British company in Hong Kong.

 This made it the first Chinese-owned bank in Hong Kong.

 At the time, all the other banks in Hong Kong were foreign, primarily British, including the locally incorporated but British-run Hongkong and Shanghai Banking
Corporation.

 Security Pacific National Bank (SPNB) bought a majority share in Bank of Canton in 1971.

 In 1988, Security Pacific succeeded in wholly acquiriing Bank of Canton, which became Security Pacific Asia Bank.

 When Bank of America acquired Security Pacific Corporation in 1993 it changed Security Pacific Asia Bank’s name to Bank of America (Asia).

 China Construction Bank acquired Bank of America (Asia) in 2006.
</item 1>

<item 2>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 2>

<item 3>
HSBC lions
Various headquarters and branch buildings of The Hongkong and Shanghai Banking Corporation and the HSBC Group, into which the Hongkong and Shanghai Banking
Corporation has evolved, feature a pair of lion sculptures.

 The HSBC lions have become distinctive landmarks in their own right in Hong Kong and Shanghai respectively, with a further pair to be found in London.
</item 3>

<item 4>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 4>

<item 5>
Bank Street (Hong Kong)
Bank Street () is a short street in Central, Hong Kong.

 It links Des Voeux Road Central to Queen's Road Central.

 It is named after its location between HSBC Building and the Bank of China Building, the former headquarters of HSBC and Bank of China (Hong Kong)
respectively.
</item 5>

<item 6>
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 6>

<item 7>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 7>

<item 8>
HSBC
HSBC Holdings PLC is a British multinational banking and financial services holding company, tracing its origin to a hong in Hong Kong.

 It is the world's seventh largest bank by total assets and the largest in Europe with total assets of US$2.374 trillion (as of December 2016).

 It was established in its present form in London in 1991 by The Hongkong and Shanghai Banking Corporation Limited to act as a new group holding company.

 The origins of the bank lie mainly in Hong Kong and to a lesser extent in Shanghai, where branches were first opened in 1865.

 The HSBC name is derived from the initials of the Hongkong and Shanghai Banking Corporation.

 The company was first formally incorporated in 1866.

 The company continues to see both the United Kingdom and Hong Kong as its "home markets".
</item 8>

<item 9>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 9>

<item 10>
HSBC Building, the Bund
The HSBC Building (also known as the Municipal Government Building) is a six-floor neo-classical building in the Bund area of Shanghai, China.

 It served as the headquarters of the Shanghai branch of The Hongkong and Shanghai Banking Corporation from 1923 to 1955, and currently houses the Shanghai
Pudong Development Bank.

 The building is situated at number 12, the Bund.

 Designed by the British architecture firm Palmer & Turner Architects and Surveyors, construction of the building lasted from 5 May 1921 to 23 June 1923.
</item 10>
=== END ITEM id=833cb3597da127b2 ===

=== ITEM id=a473702327afc9a9 ===
Gold sentences and facts:
Sentence 0: Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and Randall Counties, Texas, United States.
  Fact 0.0: Rick Husband Amarillo International Airport has the IATA code AMA.
  Fact 0.1: Rick Husband Amarillo International Airport has the ICAO code KAMA.
  Fact 0.2: Rick Husband Amarillo International Airport has the FAA LID AMA.
  Fact 0.3: Rick Husband Amarillo International Airport is a public airport.
  Fact 0.4: Rick Husband Amarillo International Airport is six miles (10 km) east of downtown Amarillo.
  Fact 0.5: Rick Husband Amarillo International Airport is in Potter and Randall Counties, Texas, United States.
Sentence 1: Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County, North Dakota.
  Fact 1.0: Grand Forks International Airport has the IATA code GFK.
  Fact 1.1: Grand Forks International Airport has the ICAO code KGFK.
  Fact 1.2: Grand Forks International Airport has the FAA LID GFK.
  Fact 1.3: Grand Forks International Airport is a public airport.
  Fact 1.4: Grand Forks International Airport is five miles (8 km) northwest of Grand Forks.
  Fact 1.5: Grand Forks International Airport is in Grand Forks County, North Dakota.

Retrieved items:
<item 1>
Rick Husband Amarillo International Airport
Rick Husband Amarillo International Airport (IATA: AMA, ICAO: KAMA, FAA LID: AMA) is a public airport six miles (10 km) east of downtown Amarillo, in Potter and
Randall Counties, Texas, United States.

 The airport was renamed in 2003 after NASA astronaut and Amarillo native Rick Husband, who died in the Space Shuttle "Columbia" disaster in February of that
year.
</item 1>

<item 2>
Emerado, North Dakota
Emerado is a city in Grand Forks County, North Dakota, United States located near Grand Forks Air Force Base.

 It is part of the "Grand Forks, ND-MN Metropolitan Statistical Area" or "Greater Grand Forks".

 The population was 414 at the 2010 census.

 Grand Forks Air Force Base is located near Emerado.
</item 2>

<item 3>
Texas Air &amp; Space Museum
The Texas Air & Space Museum is an aviation museum located near Rick Husband Amarillo International Airport in Amarillo, Texas.

 The museum displays civilian and military aircraft as well as a wide range of air and space artifacts.
</item 3>

<item 4>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 4>

<item 5>
Grand Sky
Grand Sky is an UAS (unmanned aerial systems)-specific business and aviation park located at Grand Forks Air Force Base near Grand Forks, North Dakota.

 A groundbreaking for the development was held in September 2015, making it the first UAS business and aviation park in the U.S.

 The park consists of 217 acres and includes access to the Grand Forks Air Force Base runway for large UAS flights.
</item 5>

<item 6>
Grand Forks International Airport
Grand Forks International Airport (IATA: GFK, ICAO: KGFK, FAA LID: GFK) is a public airport five miles (8 km) northwest of Grand Forks, in Grand Forks County,
North Dakota.

 GFK has no scheduled passenger flights out of the country but has an "international" title (like many other airports) because it has customs service for
arrivals from Canada and other countries.
</item 6>

<item 7>
Lea County Regional Airport
Lea County Regional Airport (IATA: HOB, ICAO: KHOB) (Lea County-Hobbs Airport) is four miles (6.4 km) west of Hobbs, in Lea County, New Mexico.

 The airport covers 898 acre and has three runways.

 It is an FAA certified commercial airport served by United Airlines' affiliate with daily regional flights.

 Lea County Regional Airport is the largest of the three airports owned and operated by Lea County Government.

 Lea County also owns and operated two general aviation airports in Lovington and Jal, New Mexico.
</item 7>

<item 8>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 8>

<item 9>
Downtown Grand Forks
Downtown Grand Forks is the original commercial center of Grand Forks, North Dakota.

 Located on the western bank of the Red River of the North, the downtown neighborhood is situated near the fork of the Red River and the Red Lake River.

 While downtown is no longer the dominant commercial area of the Greater Grand Forks community, it remains the historic center of Grand Forks.

 An 80.4 acre portion was listed on the National Register of Historic Places in 2005, as Downtown Grand Forks Historic District.

 Today, downtown Grand Forks is home to many offices, stores, restaurants, and bars.
</item 9>

<item 10>
Grand Forks International
The Grand Forks International (GFI) is an annual international invitational baseball tournament hosted at James Donaldson Park in Grand Forks, British Columbia.

 The GFI is the largest invitational baseball tournament in Canada and is a large part of the culture in Grand Forks.

 The tournament is also notable for being run entirely by volunteers.
</item 10>
=== END ITEM id=a473702327afc9a9 ===

=== ITEM id=2047987a80aa815d ===
Gold sentences and facts:
Sentence 0: Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous family.
  Fact 0.0: Homer Jay Simpson is a fictional character.
  Fact 0.1: Homer Jay Simpson is the main protagonist of the American animated television series 'The Simpsons'.
  Fact 0.2: Homer Jay Simpson is the patriarch of the Simpson family, the family the show is named after.
Sentence 1:  He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April 19, 1987.
  Fact 1.0: Homer Simpson is voiced by Dan Castellaneta.
  Fact 1.1: Homer Simpson first appeared on television along with the rest of his family.
  Fact 1.2: Homer Simpson's first television appearance was in the short 'Good Night' from 'The Tracey Ullman Show'.
  Fact 1.3: Homer Simpson's first television appearance was on April 19, 1987.
Sentence 2: Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.
  Fact 2.0: Daniel Louis Castellaneta was born October 29, 1957.
  Fact 2.1: Daniel Louis Castellaneta is an American actor.
  Fact 2.2: Daniel Louis Castellaneta is a voice actor.
  Fact 2.3: Daniel Louis Castellaneta is a comedian.
  Fact 2.4: Daniel Louis Castellaneta is a screenwriter.

Retrieved items:
<item 1>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 1>

<item 2>
Dan Castellaneta
Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.

 Noted for his long-running role as Homer Simpson on the animated television series "The Simpsons", he also voices many other characters for the show, including
Abraham "Grampa" Simpson, Barney Gumble, Krusty the Clown, Sideshow Mel, Groundskeeper Willie, Mayor Quimby and Hans Moleman.
</item 2>

<item 3>
Ned Flanders
Nedward "Ned" Flanders, Jr. is a recurring fictional character in the animated television series "The Simpsons".

 He is voiced by Harry Shearer, and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 He is the good-natured, cheery next-door neighbor to the Simpson family and is generally loathed by Homer Simpson.

 A devout Evangelical Christian with an annoyingly perfect family, he is among the friendliest and most compassionate of Springfield's residents and is
generally considered a pillar of the Springfield community.
</item 3>

<item 4>
Citizen Joe
"Citizen Joe" is the fifteenth episode for season eight of the Canadian-American military science fiction television series "Stargate SG-1".

 The episode features known voice actor Dan Castellaneta, who voices Homer Simpson in "The Simpsons".

 The episode was written by executive producer Robert C. Cooper, the episode was directed by Andy Mikita.

 The episode received a below average Nielsen household rating and received no syndication rating to compare.

 The episode got strong reviews from major media publishers worldwide.
</item 4>

<item 5>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 5>

<item 6>
Grampa Simpson
Abraham Jedediah Simpson II, often known as Grampa, is a fictional character in the animated television series "The Simpsons".

 He made his first appearance in the episode entitled "Grampa and the Kids", a Simpsons short on "The Tracey Ullman Show".

 Voiced by Dan Castellaneta, he is the father of Homer Simpson and the grandfather of Bart, Lisa and Maggie Simpson.

 In the 1000th issue of "Entertainment Weekly", Grampa was selected as the Grandpa for "The Perfect TV Family".
</item 6>

<item 7>
Homer to the Max
"Homer to the Max" is the thirteenth episode of "The Simpsons"<nowiki>'</nowiki> tenth season.

 It originally aired on the Fox network in the United States on February 7, 1999.

 In the episode, Homer discovers that a new television show, "Police Cops", has a hero also named Homer Simpson.

 He is delighted with the positive attention he receives because of his name, but when the television character is rewritten from a hero to a bumbling idiot, he
is mocked and taunted, so he changes his name to "Max Power" to rid himself of the negative attention.

 Max gains new friends, and is forced into a protest to prevent a forest from being knocked down.

 In the end, he changes his name back to Homer Simpson.
</item 7>

<item 8>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 8>

<item 9>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 9>

<item 10>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 10>
=== END ITEM id=2047987a80aa815d ===

=== ITEM id=31b5cf86541ae350 ===
Gold sentences and facts:
Sentence 0: Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.
  Fact 0.0: Ronkonkoma is a major railroad station.
  Fact 0.1: Ronkonkoma is a major transportation hub.
  Fact 0.2: Ronkonkoma is along the Main Line of the Long Island Rail Road.
  Fact 0.3: The Ronkonkoma railroad station is in Ronkonkoma, New York.
Sentence 1: Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.
  Fact 1.0: Ronkonkoma is a hamlet.
  Fact 1.1: Ronkonkoma is a census-designated place (CDP).
  Fact 1.2: Ronkonkoma is on Long Island.
  Fact 1.3: Ronkonkoma is in the Town of Islip.
  Fact 1.4: The Town of Islip is in New York, United States.

Retrieved items:
<item 1>
North Creek Railroad Station
The North Creek Railroad Station is a historic railroad station complex located at North Creek, Warren County, New York.

 The complex consists of the railroad station, the freight house, round house, turntable, and horse barn.

 The station was built in 1874 and is a simple, rectangular, gable roofed building with a broad, overhanging strut-supported roof in the Stick-Eastlake style.

 Its exterior is covered with vertical boards.
</item 1>

<item 2>
Jamestown Gateway Train Station
The Jamestown Gateway Train Station, also known as the Jamestown Erie Railroad station, and the Erie-Lackawanna Railroad Station, is a historic train station
located at Jamestown in Chautauqua County, New York.

 Although no longer an active railroad station due to a lack of passenger service in the area after a restoration done in 2011 the building currently serves as
a bus transportation center and community space for Jamestown.

 The first train arrived at Jamestown on August 25, 1860 as part of the Atlantic and Great Western Railroad.
</item 2>

<item 3>
Lacona Railroad Station and Depot
Lacona Railroad Station and Depot, also known as New York Central and Hudson River Railroad Station and Depot, is a historic railway depot located at Lacona in
Oswego County, New York.

 It was built in 1891 by the New York Central and Hudson River Railroad.

 It is a small rectangular, one story, gable ended structure.
</item 3>

<item 4>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 4>

<item 5>
Ridgefield Park station
Ridgefield Park Station, also known as West Shore Station, was railroad station in Ridgefield Park, New Jersey at the foot of Mount Vernon Street served by the
New York, Susquehanna and Western Railroad (NYSW) and the West Shore Railroad, a division of New York Central (NYCRR) The New York, Ontario and Western Railway
(NYO&W) had running rights along the West Shore and sometimes stopped at Ridgefield Park.

 First opened in 1883 it was one of three passenger stations in the village, the others being the Little Ferry Station to the south and Westview Station to the
north.

 The station house, built at a cost $100,000 opened in 1927.

 Southbound service crossed Overpeck Creek and continued to terminals on the Hudson River waterfront where there was connecting ferry service across the Hudson
River to Manhattan.

 Northbound near Bogota the parallel NYSW and West Shore lines diverge and continue into northern New Jersey, Pennsylviania, and upstate New York.

 Passenger service ended in 1966.
</item 5>

<item 6>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 6>

<item 7>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 7>

<item 8>
Ronkonkoma (LIRR station)
Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.

 The station is the eastern terminus of the Ronkonkoma Branch and the western terminus of the out Branch.
</item 8>

<item 9>
Long Island Electric Railway
The Long Island Electric Railway was a streetcar company operating in Brooklyn, Queens, and Nassau County, New York, United States between 1894 and 1926.

 The company was partially owned by the Long Island Consolidated Electric Companies, a holding company for the Long Island Rail Road and partially by August
Belmont and the Interborough Rapid Transit Company.

 It connected the east end of the Fulton Street El at Crescent Street station in City Line, Brooklyn with Jamaica, Queens, and ran from there to the Nassau
County line at Queens Village and to Far Rockaway, Queens via Nassau County.

 It also had a connection to Belmont Park.

 The New York and Long Island Traction Company used trackage rights over its line from Crescent Street to Queens Village.
</item 9>

<item 10>
Lake Worth station
Lake Worth is a Tri-Rail commuter rail station in Lake Worth, Florida, at the confluence of Lake Worth Road (SR 802) and Interstate 95.

 Opening to service January 9, 1989, parking is available at this station, all of which is beneath I-95 on the south side of Lake Worth Road.
</item 10>
=== END ITEM id=31b5cf86541ae350 ===

=== ITEM id=459b57ccf417d247 ===
Gold sentences and facts:
Sentence 0: Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
  Fact 0.0: Hänt Extra is a Swedish language magazine.
  Fact 0.1: Hänt Extra is a weekly magazine.
  Fact 0.2: Hänt Extra is a celebrity and youth magazine.
  Fact 0.3: Hänt Extra is published by Aller Media.
  Fact 0.4: Hänt Extra is published in Stockholm, Sweden.
Sentence 1: Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
  Fact 1.0: Tea Moderna is a Macedonian woman's weekly magazine.
  Fact 1.1: Tea Moderna's name in Macedonian is Теа Модерна.
  Fact 1.2: Tea Moderna is a magazine for women.

Retrieved items:
<item 1>
Cult MTL
Cult MTL is an English language arts, culture and news website and monthly print publication, based in Montreal, Quebec, Canada.

 Its first print edition appeared on 7 September 2012.

 It was created only a few months after Montreal's last English-language alternative weekly, "Montreal Mirror", was unceremoniously closed by its parent
company, Quebecor.

 The founding editor's of "Cult MTL" were also involved with the "Mirror".

 In August 2013 the print version of the magazine was started.
</item 1>

<item 2>
Micom Film and TV Productions
MICOM Film and TV Productions was established in 1989 by Sreco Mihelcic in a Former Yugoslavian republic Slovenia.

 In the early 90' company joined two sons, Matjaz Mihelcic - Program director, Producer and Vasja Mihelcic - technical director and Executive Producer.

 MICOM was in 1990s the leading Slovenian creative house, producing TV shows for youth, popular entertainment TV shows and discovered many new TV concepts.

 Micom discovered many talents, different creative people and gave them possibility to express themselves.

 The most popular TV shows were: Lahkih nog naokrog (educational youth TV series), Cari zacimb (entertaining cooking TV series), Razpoke v casu (fantasy TV
series).

 They also produced documentary movies.

 One of them got in 1996 internationally release - Fight for the river and it's distributed by International Historic Films.

.

 In the beginning of 2000 company joined Marjana Mihelcic.

 It was a family based company with long tradition in show and entertainment business.
</item 2>

<item 3>
Hekmat
Hekmat (Persian: حكمت‎ ‎  –"Wisdom") was the first Persian-language newspaper to be published in Egypt, as well as the first Persian journal to be published in
an Arab country.

 Founded and managed by the Iranian expat Mohammad-Mahdi Tabrizi (died 1914), a physician by profession, it was published from 20 September 1892 until 30 May
1911.

 "Hekmat" carried mostly news, but also feature a variety of articles on political and social issues.

 Despite being published in an "Arab country", it avoided the usage of Arabic terms and compoun words as well as Arabicized forms of non-Arab words throughouts
its texts.
</item 3>

<item 4>
Hänt Extra
Hänt Extra is a Swedish language weekly celebrity and youth magazine published by Aller Media in Stockholm, Sweden.
</item 4>

<item 5>
Hänt Bild
Hänt Bild is a celebrity magazine published in Sweden.

 The magazine was started in 2003.

 It is part of Aller Media AB.

 It is published on a weekly basis.
</item 5>

<item 6>
Makedonsko Sonce
Makedonsko Sonce (Macedonian: ; English: Macedonian Sun ) is a weekly magazine published in the Republic of Macedonia.

 The title means "Macedonian Sun", referring to the Vergina Sun, which is used in the logo of the magazine.

 It was established by Gjorgija (George) Atanasoski and the first edition was published on June 24, 1994.

 It describes its own political stance as affirming "Macedonian national values, not only on territory of Macedonia but also in the other parts of ethnic
Macedonia".
</item 6>

<item 7>
Nikola Mladenov
Nikola Mladenov (Macedonian: Никола Младенов ; 10 March 1964 – 26 March 2013) was a Macedonian journalist, actor, founder and managing editor of the weekly
newspaper "Fokus", widely regarded as one of the most reputable journalists in Macedonia, who along with his colleagues from the newspaper "Mlad borec" is
considered to have laid the basis for the pluralistic principles in the Macedonian journalism.

 The "Award for best research story of the year – Nikola Mladenov", which is awarded by the Macedonian Institute for Media, is named after him.
</item 7>

<item 8>
Tea Moderna (magazine)
Tea Moderna (Macedonian: Теа Модерна ) is a Macedonian woman's weekly magazine women.
</item 8>

<item 9>
Pravda Pyat
Pravda Pyat (Russian: правда пять or "Truth Five") is a weekly Russian tabloid news publication that was a spin-off from "Pravda".

 It was founded by Greek entrepreneurs Christos Giannikos and Fyodoros Giannikos of Pravda International in 1996.

 The magazine replaced "Pravda" when that publication ceased operations.

 Its intended audience was younger readers, and coverage was more sensationalistic than "Pravda", focusing on crime and scandals.
</item 9>

<item 10>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 10>
=== END ITEM id=459b57ccf417d247 ===
