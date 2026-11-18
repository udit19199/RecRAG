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

=== ITEM id=5db431a456507302 ===
Gold sentences and facts:
Sentence 0: Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.
  Fact 0.0: Ted Kooshian was born October 8, 1961.
  Fact 0.1: Ted Kooshian is an American jazz pianist.
  Fact 0.2: Ted Kooshian is an American jazz keyboardist.
  Fact 0.3: Ted Kooshian is an American jazz composer.
  Fact 0.4: Ted Kooshian has performed with Aretha Franklin.
  Fact 0.5: Ted Kooshian has performed with Chuck Berry.
  Fact 0.6: Ted Kooshian has performed with Marvin Hamlisch.
  Fact 0.7: Ted Kooshian has performed with Sarah Brightman.
  Fact 0.8: Ted Kooshian has performed with Il Divo.
  Fact 0.9: Ted Kooshian has performed with Blood, Sweat, and Tears.
Sentence 1:  Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
  Fact 1.0: Marvin Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.
Sentence 2:  This collection of all four is referred to as an "EGOT".
  Fact 2.0: The collection of all four awards (Emmy, Grammy, Oscar and Tony) is referred to as an 'EGOT'.

Retrieved items:
<item 1>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 1>

<item 2>
List of James Tiptree Jr. Award winners
The following is a list of winners and shortlisted works of the James Tiptree Jr.

 Award, an annual literary prize for works of science fiction or fantasy that expand or explore one's understanding of gender.

 It was initiated in February 1991 by science fiction authors Pat Murphy and Karen Joy Fowler, and is awarded and discussed yearly at WisCon.
</item 2>

<item 3>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 3>

<item 4>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 4>

<item 5>
Tatsuhiro Oshiro
Tatsuhiro Ōshiro (大城 立裕 , Ōshiro Tatsuhiro , born 19 September 1925) is a Japanese novelist and playwright.

 He was awarded the Akutagawa Prize in 1967 for his novella of the same year, "The Cocktail Party", which has been adapted for the stage and made into a film.

 Ōshiro has also been an innovator of the traditional Ryukyuan narrative dance form known as kumi odori.

 Having added twenty new pieces to the repertoire, Ōshiro is credited as having "single-handedly revived the genre that originated in the 18th century" by
incorporating Okinawa shibai (dramas in the Okinawan language) and distinctive rhythms to construct a fluid, hybrid cultural identity.

 His writings have been noted for making Okinawan culture and history accessible to Japanese readership, while his more popular works have been critically
praised for "offering an acute perspective on the psychological and moral implications of war and military occupation."
</item 5>

<item 6>
Premio Lo Nuestro 2003
In the Regional Mexican, Pilar Montenegro took three awards for Regional Mexican Song of the Year, Pop Song of the Year ("Quitame Ese Hombre"), and for Best
Regional Mexican Female Artist.

 In the tropical genre, Celia Cruz took home with four great awards of the night for Best Salsa Performance, Best Tropical Female Artist, Best Tropical Song of
the year and Tropical Album of the Year.

 At the night, the greatest performance was a medley of top Latin hits from the last 15 years, performed by the artists that made them famous, including Vikki
Carr, Son by Four, Los Ilegales, La Mafia, Luis Enrique, Wilfrido Vargas and Olga Tañón.

 There was a great tribute to Celia Cruz by the world-famous salsa group "Fania All-Stars", of which Cruz was a member during the 1970s, reunited for an
exclusive performance that rocked the house.
</item 6>

<item 7>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 7>

<item 8>
Leonard Cohen
Leonard Norman Cohen {'1': ", '2': ", '3': ", '4': "} (September 21, 1934 – November 7, 2016) was a Canadian singer, songwriter, musician, poet, novelist, and
painter.

 His work explored religion, politics, isolation, sexuality, and personal relationships.

 Cohen was inducted into the Canadian Music Hall of Fame, the Canadian Songwriters Hall of Fame, and the Rock and Roll Hall of Fame.

 He was a Companion of the Order of Canada, the nation's highest civilian honor.

 In 2011, Cohen received one of the Prince of Asturias Awards for literature and the ninth Glenn Gould Prize.
</item 8>

<item 9>
Triple Crown of Acting
The Triple Crown of Acting is a term used in the American entertainment industry to describe actors who have won a competitive Academy Award, Emmy Award, and
Tony Award in the acting categories.

 23 people have achieved the triple crown of acting (14 women, 9 men).

 Helen Hayes' Emmy Award win on February 5, 1953, made her the first person to achieve the triple crown.

 Thomas Mitchell became the first man to achieve the triple crown with his Tony Award win later the same year on March 29, 1953.

 Hayes and Rita Moreno are the only triple crown winners in competitive acting categories who have also won a Grammy Award to complete the EGOT.
</item 9>

<item 10>
List of awards and nominations received by Kim Tae-yeon
List of awards and nominations received by Kim Tae-yeon
</item 10>
=== END ITEM id=5db431a456507302 ===

=== ITEM id=062c4f3fa3ed6d9a ===
Gold sentences and facts:
Sentence 0: Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and songwriter.
  Fact 0.0: Toshimitsu Deyama was born October 10, 1965 in Tateyama, Chiba.
  Fact 0.1: Toshimitsu Deyama's Japanese name is 出山利三.
  Fact 0.2: Toshimitsu Deyama's name is also written as Deyama Toshimitsu.
  Fact 0.3: Toshimitsu Deyama is known exclusively by his stage name Toshi.
  Fact 0.4: Toshimitsu Deyama is a Japanese singer and songwriter.
Sentence 1:  He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.
  Fact 1.0: Toshimitsu Deyama is best known as the lead vocalist of the heavy metal band X Japan.
  Fact 1.1: Toshimitsu Deyama is a co-founder of X Japan.
  Fact 1.2: X Japan disbanded in 1997.
Sentence 2: Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.
  Fact 2.0: Warrel Dane was born March 7, 1969.
  Fact 2.1: Warrel Dane is the lead singer of the American power metal band Sanctuary.

Retrieved items:
<item 1>
Sanctuary (band)
Sanctuary is an American heavy metal band founded in Seattle in 1985.

 They split up in 1992, but reformed 18 years later.

 The band consists of Warrel Dane (vocals), Lenny Rutledge (guitar), Nick Cordle (guitar), George Hernandez (bass), and Dave Budbill (drums).

 To date, they have released four studio albums and one live EP.
</item 1>

<item 2>
Into the Mirror Live
Into The Mirror Live / Black Reflections is live promotional EP of the band Sanctuary recorded on May 12, 1990 at The Country Club in Reseda, California, during
their "Into the Mirror Black" tour.

 Approximately 1000 CD copies of this were made and less than 500 were ever distributed to radio stations and record stores.

 Because of contractual problems with Epic Records, the full version was never released for public consumption.

 Epic Records destroyed 500+ copies of this CD.

 Warrel Dane has stated on several occasions that he has the master tapes and copyrights of the full show and intended to have it finally released.

 To date that has not happened.

 This CD has been heavily pirated over the years as a 2-on-1 CD with the Satan's Host "Metal From Hell" album on the notorious Reborn Classics record label.
</item 2>

<item 3>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 3>

<item 4>
Nevermore (album)
Nevermore is the eponymous debut studio album by American heavy metal band Nevermore, released on February 14, 1995 by Century Media Records.

 It was singer Warrel Dane and bassist Jim Sheppard's first release after departing from their previous band, Sanctuary, in 1994.

 Having recruited drummer Van Williams and former Sanctuary touring guitarist Jeff Loomis, they formed the band Nevermore and began anew.
</item 4>

<item 5>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 5>

<item 6>
Praises to the War Machine
Praises to the War Machine is the debut and only solo release/album by heavy metal vocalist Warrel Dane from Nevermore, released on April 25, 2008.

 The album was produced by Peter Wichers, who was tapped by Dane to work on the album while Wichers was still a member of Soilwork.

 After leaving that band in 2007, Wichers co-wrote the album with Dane and played on eight of its tracks.

 Soilwork drummer Dirk Verbeuren and former Himsa guitarist Matt Wicklund also played on the album.

 Nevermore guitarist Jeff Loomis and the band's touring guitarist at the time Chris Broderick make guest appearances, along with James Murphy, who had performed
on the Nevermore album "This Godless Endeavor".

 "Praises to the War Machine" includes two cover songs, The Sisters of Mercy's "Lucretia My Reflection" and "Patterns" by Paul Simon.

 Dane had previously covered a Simon song on Nevermore's album "Dead Heart in a Dead World".
</item 6>

<item 7>
The Halo Method
The Halo Method Is an American rock supergroup, formed in 2012 in Hollywood, California.

 It consisted of guitarist Ben Moody (We Are the Fallen, ex-Evanescence), drummer Miles McPherson, vocalist Lukas Rossi (ex-Rock Star Supernova, ex-Daylight
Division) and bassist Josh Newell (ex-In This Moment).
</item 7>

<item 8>
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 8>

<item 9>
Toshi (musician)
Toshimitsu Deyama (出山利三 , Deyama Toshimitsu , born October 10, 1965 in Tateyama, Chiba) , known exclusively by his stage name Toshi, is a Japanese singer and
songwriter.

 He is best known as the lead vocalist and a co-founder of the heavy metal band X Japan, which disbanded in 1997.

 Toshi then went on to have an extensive solo career.

 In 2007, X Japan reunited and began an ongoing world tour in 2010.

 His distinctive, impassioned tenor vocals have been compared to those of Steve Perry.
</item 9>

<item 10>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 10>
=== END ITEM id=062c4f3fa3ed6d9a ===

=== ITEM id=cfb0d06892b96f2b ===
Gold sentences and facts:
Sentence 0: Dean Mills Reservoir is a small water reservoir to the north of the town of Bolton, Greater Manchester located high on the slopes of Winter Hill.
  Fact 0.0: Dean Mills Reservoir is a small water reservoir.
  Fact 0.1: Dean Mills Reservoir is north of the town of Bolton, Greater Manchester.
  Fact 0.2: Dean Mills Reservoir is located high on the slopes of Winter Hill.
Sentence 1:  It is located on Rivington Moor, Chorley and is 1496 ft high.
  Fact 1.0: Dean Mills Reservoir is located on Rivington Moor, Chorley.
  Fact 1.1: Dean Mills Reservoir is 1496 ft high.

Retrieved items:
<item 1>
Daniel Mills (biologist)
Daniel Simon Mills, FRCVS (born 21 Aug 1966) is an English biologist and the UK's first Professor of Veterinary Behavioural Medicine based at the University of
Lincoln, United Kingdom.

 He attended St Edmund's College, Ware before studying at the University of Bristol where he received his BVSc degree and completed his Ph.D. degree in animal
behaviour from De Montfort University.

 After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and
subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.

 His research has focused on the assessment of emotion in companion animals and the use of semiochemicals to manage their problem behaviour (pheromonatherapy).
</item 1>

<item 2>
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building in Preston. Lichfield’s Minster Pool, formed in the 11th century to
power a mill and defend the Cathedral Close, became a public amenity after the mill’s demolition in 1856. Cheshire’s adjacent medieval hunting forests of Mara
and Mondrem covered over 60 square miles and held boar and deer. Dean Mills Reservoir lies high on Winter Hill, north of Bolton.
</item 2>

<item 3>
Dean Mills Reservoir
Dean Mills Reservoir is a small water reservoir to the north of the town of Bolton, Greater Manchester located high on the slopes of Winter Hill.
</item 3>

<item 4>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 4>

<item 5>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 5>

<item 6>
Minster Pool
Minster Pool is a reservoir located between Bird Street and Dam Street in the heart of the city of Lichfield, Staffordshire in the United Kingdom.

 The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.

 The pool was originally formed in the 11th century when a boggy stream was dammed at its eastern end to drive a mill on Dam Street.

 The pool was used as a mill pond and fishery until 1856 when the mill was demolished; it has since been retained for public amenity.
</item 6>

<item 7>
Fisher Hill Reservoir
The Fisher Hill Reservoir and Gatehouse are a historic element of the public water supply for the Greater Boston area.

 The reservoir was located on Fisher Road between Hyslop and Channing Roads in Brookline, Massachusetts, and is now the site of Fisher Hill Reservoir Park.

 It was built in 1886-87 as an early component of the Boston Water Board's expansion of its high service system.

 The gatehouse may have been designed by Arthur Vinal, who also designed the high pumping station at Chestnut Hill Reservoir.

 It is a two-story Richardsonian Romanesque structure, with its first floor finished in stone and its second in brick.

 Brownstone trim is used on the windows and corner quoins, and the voussoirs which form the arches on the first floor.

 There are pipes to the reservoir and down to Chestnut Hill, and gates for controlling access to local the distribution network.

 The building was taken out of service in the 1950s.
</item 7>

<item 8>
Winter Hill (North West England)
Winter Hill is a hill on the border of the boroughs of Chorley, Blackburn with Darwen and Bolton, in North West England.

 It is located on Rivington Moor, Chorley and is 1496 ft high.

 Part of the West Pennine Moors, it is a popular walking area, and has been the site of mining activity, aeroplane disasters and murders.
</item 8>

<item 9>
Arnold Mills Historic District
Arnold Mills Historic District is a historic district encompassing a modest 19th-century mill village in eastern Cumberland, Rhode Island.

 The district lies along the Nate Whipple Highway (Rhode Island Route 120) and Sneech Pond Road, south of the Arnold Mills Reservoir.

 Sneech Pond Road was formerly the major east-west highway through the area prior to the construction of the Nate Whipple Highway in the 1960s.

 The Arnold Mills village is in part bisected by Abbott Run, the stream which serves as the outlet of the reservoir; Sneech Pond Road crosses the run on an
early 20th-century steel Pratt pony truss, now closed to vehicular traffic.

 The houses along this road generally date from the late 18th to mid-19th century, and mainly reflect Federal and Greek Revival styling.

 The most prominent structure in the district is the Arnold Mills United Methodist Church, located at the western end of the district on Nate Whipple Highway;
it was built 1825-27 and remodeled in 1846.
</item 9>

<item 10>
Toorourrong Reservoir
Toorourrong Reservoir is a small water supply reservoir located on the southern slopes of the Great Dividing Range approximately 40 km north of Melbourne,
Victoria, Australia.

 The reservoir is formed by the Toorourrong Dam across the Plenty River, and an interbasin transfer.

 The dam is operated by Melbourne Water and the reservoir forms part of the Melbourne water supply system.

 Water from the Toorourrong Reservoir flows by aqueduct to the Yan Yean Reservoir.
</item 10>
=== END ITEM id=cfb0d06892b96f2b ===

=== ITEM id=33c11c85e6e37a8c ===
Gold sentences and facts:
Sentence 0: Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to become lead singer of the hard rock supergroup "Rock Star Supernova".
  Fact 0.0: Lukas Rossi was born on December 21, 1976.
  Fact 0.1: Lukas Rossi is a Canadian rock musician.
  Fact 0.2: Lukas Rossi was the winner of a CBS Television reality series.
  Fact 0.3: That CBS Television reality series was a televised audition contest to become lead singer of the hard rock supergroup 'Rock Star Supernova'.
Sentence 1: Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and founding members of the rock band Chicago.
  Fact 1.0: Terry Alan Kath was born on January 31, 1946.
  Fact 1.1: Terry Alan Kath died on January 23, 1978.
  Fact 1.2: Terry Alan Kath was an American musician.
  Fact 1.3: Terry Alan Kath was an American songwriter.
  Fact 1.4: Terry Alan Kath was best known as the original guitarist of the rock band Chicago.
  Fact 1.5: Terry Alan Kath was one of the lead singers of the rock band Chicago.
  Fact 1.6: Terry Alan Kath was a founding member of the rock band Chicago.

Retrieved items:
<item 1>
Graph query row:
person: Terry Alan Kath
nationality: American
</item 1>

<item 2>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 2>

<item 3>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 3>

<item 4>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 4>

<item 5>
Lukas Rossi
Lukas Rossi (born December 21, 1976) is a Canadian rock musician and was the winner of the CBS Television reality series "" - a televised audition contest to
become lead singer of the hard rock supergroup "Rock Star Supernova".
</item 5>

<item 6>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 6>

<item 7>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 7>

<item 8>
Graph query row:
person: Lukas Rossi
nationality: Canadian
</item 8>

<item 9>
Terry Kath
Terry Alan Kath (January 31, 1946 – January 23, 1978) was an American musician and songwriter, best known as the original guitarist, one of the lead singers and
founding members of the rock band Chicago.

 He has been praised by the band for his guitar skills and Ray Charles-influenced vocal style.
</item 9>

<item 10>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 10>
=== END ITEM id=33c11c85e6e37a8c ===

=== ITEM id=471bfce2163efb29 ===
Gold sentences and facts:
Sentence 0:  One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.
  Fact 0.0: A nunatak was first roughly mapped by the Second German Antarctic Expedition of 1911–12
  Fact 0.1: The Second German Antarctic Expedition of 1911–12 was under Wilhelm Filchner
  Fact 0.2: The Second German Antarctic Expedition named the nunatak Moltke Nunatak
Sentence 1:  He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.
  Fact 1.0: The nunatak was named for General Helmuth von Moltke
  Fact 1.1: General Helmuth von Moltke was Chief of the German General Staff
  Fact 1.2: General Helmuth von Moltke was Secretary of State for Home Affairs
Sentence 2: Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal) Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.
  Fact 2.0: Helmuth Johann Ludwig von Moltke was born 23 May 1848
  Fact 2.1: Helmuth Johann Ludwig von Moltke died 18 June 1916
  Fact 2.2: Helmuth Johann Ludwig von Moltke is also known as Moltke the Younger
  Fact 2.3: Helmuth Johann Ludwig von Moltke was a nephew of Generalfeldmarschall (Field Marshal) Helmuth Karl Bernhard von Moltke
  Fact 2.4: Helmuth Johann Ludwig von Moltke served as Chief of the German General Staff from 1906 to 1914

Retrieved items:
<item 1>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 1>

<item 2>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 2>

<item 3>
Admirals Nunatak
Admirals Nunatak ( ) is a nunatak rising to 925 m on the upper Uranus Glacier, central Alexander Island, Antarctica.

 The name originates from dog teams named "The Admirals" that served at various British stations in Antarctica, 1952–94, and honors the loyal service of all
Falkland Islands Dependencies Survey/BAS sled dogs.

 The nunatak appears to have some relation to Huns Nunatak which lies about 3.7 mi northeast of Admirals Nunatak.
</item 3>

<item 4>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 4>

<item 5>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 5>

<item 6>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 6>

<item 7>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 7>

<item 8>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 8>

<item 9>
Moltke Nunataks
The Moltke Nunataks ( ) are a chain of north–south trending nunataks close to the northeastern end of the Filchner Ice Shelf, Antarctica.

 One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.

 He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.

 Surveys during the mid-1950s by British, Argentine and United States expeditions indicate that a group of four or five nunataks exist in the area.
</item 9>

<item 10>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 10>
=== END ITEM id=471bfce2163efb29 ===

=== ITEM id=8807a797df26345b ===
Gold sentences and facts:
Sentence 0:  The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.
  Fact 0.0: The pool lies directly south of Lichfield Cathedral.
  Fact 0.1: The pool has historically been important to the defence of the Cathedral Close.
Sentence 1:  The Diocese of Lichfield covers all of Staffordshire, much of Shropshire and part of the Black Country and West Midlands.
  Fact 1.0: The Diocese of Lichfield covers all of Staffordshire.
  Fact 1.1: The Diocese of Lichfield covers much of Shropshire.
  Fact 1.2: The Diocese of Lichfield covers part of the Black Country.
  Fact 1.3: The Diocese of Lichfield covers part of the West Midlands.

Retrieved items:
<item 1>
The West Midlands railway-station list covers stations with regular services in the county and some in TfWM’s Network West Midlands area. Dudley Zoo, opened in
1937 at Dudley Castle, is a 40-acre charity-run zoo. Walsall, Tipton and Brierley Hill are Black Country towns; Brierley Hill’s steelworks site became Merry
Hill shopping centre. Dudley also hosts the 105,000 m² Black Country Living Museum. The remaining entries describe Clayton Brook and
</item 1>

<item 2>
Walsall
Walsall ( ) is an industrial town in the West Midlands of England.

 It is located 8 miles north-west of the City of Birmingham and 6 miles east of the City of Wolverhampton.

 Historically a part of Staffordshire, Walsall is a component area of the West Midlands conurbation, and part of the Black Country.
</item 2>

<item 3>
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building in Preston. Lichfield’s Minster Pool, formed in the 11th century to
power a mill and defend the Cathedral Close, became a public amenity after the mill’s demolition in 1856. Cheshire’s adjacent medieval hunting forests of Mara
and Mondrem covered over 60 square miles and held boar and deer. Dean Mills Reservoir lies high on Winter Hill, north of Bolton.
</item 3>

<item 4>
List of railway stations in the West Midlands
This is a list of railway stations within the West Midlands, a metropolitan county in central England which includes the cities of Birmingham, Coventry and
Wolverhampton.

 It includes all railway stations in the West Midlands that currently have regular timetabled train services, as well as certain stations outside the county
which are within the area supported by Transport for West Midlands (TfWM), formerly known as Centro.

 Transport within the West Midlands is subsidised by TfWM, who since 2006 have used the brand name Network West Midlands to demonstrate the 'joined-up' nature
of the regions bus and rail networks.
</item 4>

<item 5>
Brierley Hill
Brierley Hill is a small town and electoral ward of the Dudley Metropolitan Borough, in the West Midlands of England, and is situated approximately 2.5 miles
south of central Dudley and 2 miles north of Stourbridge.

 Part of the Black Country, and in a heavily industrialised area of the Dudley Borough, it has a population of 13,935 at the 2011 census, and is best known for
glass and steel manufacturing, although the industry has declined considerably since the 1970s.

 One of the largest factories in the area was the Round Oak Steelworks, which was closed down and redeveloped in the 1980s to become the Merry Hill Shopping
Centre.

 Brierley Hill was originally in Staffordshire, but is now part of the West Midlands metropolitan county since its creation in 1974.
</item 5>

<item 6>
Black Country Living Museum
The Black Country Living Museum (formerly The Black Country Museum) is an open-air museum of rebuilt historic buildings in Dudley in the West Midlands of
England.

 It is located in the centre of the Black Country, 10 miles west of Birmingham.

 The museum occupies 105000 m2 of former industrial land partly reclaimed from a former railway goods yard, disused lime kilns, canal arm and former coal pits.
</item 6>

<item 7>
Black Country Rangers F.C.
Black Country Rangers Football Club is a football club based in Rowley Regis in the West Midlands.

 They are currently members of the West Midlands (Regional) League Premier Division and play at the Beeches in Cradley, groundsharing with Cradley Town.
</item 7>

<item 8>
Tipton
Tipton is a town in the borough of Sandwell, West Midlands, England, with a population of around 38,777 at the 2011 UK Census.

 Tipton is located about halfway between Birmingham and Wolverhampton.

 It is a part of the West Midlands conurbation and is a part of the Black Country.
</item 8>

<item 9>
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
</item 9>

<item 10>
Minster Pool
Minster Pool is a reservoir located between Bird Street and Dam Street in the heart of the city of Lichfield, Staffordshire in the United Kingdom.

 The pool lies directly south of Lichfield Cathedral and historically has been important to the defence of the Cathedral Close.

 The pool was originally formed in the 11th century when a boggy stream was dammed at its eastern end to drive a mill on Dam Street.

 The pool was used as a mill pond and fishery until 1856 when the mill was demolished; it has since been retained for public amenity.
</item 10>
=== END ITEM id=8807a797df26345b ===

=== ITEM id=d9b18c0a37598ff7 ===
Gold sentences and facts:
Sentence 0: Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
  Fact 0.0: Javier Alberto Frana was born on 25 December 1966.
  Fact 0.1: Javier Alberto Frana is a former tennis player.
  Fact 0.2: Javier Alberto Frana is from Argentina.
  Fact 0.3: Javier Alberto Frana is a current tennis commentator.
  Fact 0.4: Javier Alberto Frana is a tennis commentator for ESPN Latin America.
Sentence 1: Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and the U.S. National Championships.
  Fact 1.0: Thomaz Koch was born on May 11, 1945.
  Fact 1.1: Thomaz Koch was born in Porto Alegre.
  Fact 1.2: Thomaz Koch is a left-handed former tennis player.
  Fact 1.3: Thomaz Koch is from Brazil.
  Fact 1.4: Thomaz Koch was a quarter-finalist at the French Open.
  Fact 1.5: Thomaz Koch was a quarter-finalist at Wimbledon.
  Fact 1.6: Thomaz Koch was a quarter-finalist at the U.S. National Championships.

Retrieved items:
<item 1>
1996 IPB Czech Indoor – Doubles
Jonas Björkman and Javier Frana was the defending champions but did not compete that year.
</item 1>

<item 2>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 2>

<item 3>
1993 ATP Buenos Aires – Doubles
Pablo Albano and Javier Frana were the defending champions, but lost in the quarterfinals this year.
</item 3>

<item 4>
Thomaz Koch
Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and
the U.S. National Championships.
</item 4>

<item 5>
Franz Josef Och
Franz Josef Och (November 2, 1971) is a German computer scientist.

 He is currently working at Grail (an Illumina company) as Head of Data Science.

 Prior to this he was Chief Data Scientist at Human Longevity Inc., and earlier worked for Google as a Distinguished Research Scientist and head of machine
translation based at Google's Mountain View, California, headquarters south of San Francisco.
</item 5>

<item 6>
Javier Frana
Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
</item 6>

<item 7>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 7>

<item 8>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 8>

<item 9>
1995 Nottingham Open – Singles
Javier Frana defeated Todd Woodbridge 7–6, 6–3 in the final to secure the title.
</item 9>

<item 10>
1994 Movistar Open – Singles
Javier Frana was the defending champion but lost in the quarterfinals to Alberto Berasategui.
</item 10>
=== END ITEM id=d9b18c0a37598ff7 ===

=== ITEM id=47aaac018c94d06d ===
Gold sentences and facts:
Sentence 0: The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.
  Fact 0.0: The Cathedral Church of the Holy Spirit, Guildford is an Anglican cathedral.
  Fact 0.1: The Cathedral Church of the Holy Spirit, Guildford is located at Guildford, Surrey, England.
Sentence 1: St Paul's Cathedral, London, is an Anglican cathedral, the seat of the Bishop of London and the mother church of the Diocese of London.
  Fact 1.0: St Paul's Cathedral, London is an Anglican cathedral.
  Fact 1.1: St Paul's Cathedral, London is the seat of the Bishop of London.
  Fact 1.2: St Paul's Cathedral, London is the mother church of the Diocese of London.

Retrieved items:
<item 1>
Chelmsford Cathedral
Chelmsford Cathedral in the city of Chelmsford, Essex, England, is dedicated to St Mary the Virgin, St Peter and St Cedd.

 It became a cathedral when the Anglican Diocese of Chelmsford was created in 1914 and is the seat of the Bishop of Chelmsford.
</item 1>

<item 2>
Exterior sculpture of Guildford Cathedral
The Exterior sculpture of Guildford Cathedral provides many artistic features, including sculptures, engravings and more by some of England’s finest sculptors
and craftsmen of the 1950s and 1960s.

 The people who worked on the cathedral include: Edward Maufe, Alan Collins, Vernon Hill, Eric Gill, John Hutton, Dennis Huntley and others.
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
St Paul's Cathedral, Rockhampton
St Paul's Anglican Cathedral is a heritage-listed cathedral at 89 William Street, Rockhampton, Rockhampton Region, Queensland, Australia.

 It was designed principally by Annersley Voysey and built from 1883 to 1953.

 It is also known as St Paul's Church of England and St Paul's Church of England Cathedral.

 It was added to the Queensland Heritage Register on 21 October 1992.

 The associated hall and offices were listed on the Queensland Heritage Register on 23 June 2000.
</item 4>

<item 5>
Dean of Guildford
The Dean of Guildford is the head ("primus inter pares" – first among equals) and chair of the chapter of canons, the ruling body of Guildford Cathedral.

 The dean and chapter are based at the "Cathedral Church of the Holy Spirit" in Guildford.

 The cathedral is the mother church of the Diocese of Guildford and seat of the Bishop of Guildford.

 The current dean is Dianna Gwilliams, who was installed on 15 September 2013.
</item 5>

<item 6>
St Philip's Cathedral, Birmingham
The Cathedral Church of Saint Philip is the Church of England cathedral and the seat of the Bishop of Birmingham.

 Built as a parish church and consecrated in 1715, St Philip's became the cathedral of the newly formed Diocese of Birmingham in 1905.

 St Philip's was built in the early 18th century in the Baroque style by Thomas Archer and is located on Colmore Row, Birmingham, England.

 The cathedral is a Grade I listed building.

 St Philip's is the third smallest cathedral in England after Derby and Chelmsford.
</item 6>

<item 7>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 7>

<item 8>
St Paul's Cathedral
St Paul's Cathedral, London, is an Anglican cathedral, the seat of the Bishop of London and the mother church of the Diocese of London.

 It sits on Ludgate Hill at the highest point of the City of London and is a Grade 1 listed building.

 Its dedication to Paul the Apostle dates back to the original church on this site, founded in AD 604.

 The present cathedral, dating from the late 17th century, was designed in the English Baroque style by Sir Christopher Wren.

 Its construction, completed in Wren's lifetime, was part of a major rebuilding programme in the City after the Great Fire of London.
</item 8>

<item 9>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 9>

<item 10>
Guildford Cathedral
The Cathedral Church of the Holy Spirit, Guildford is the Anglican cathedral at Guildford, Surrey, England.

 The Lord Onslow donated the land on which the Cathedral was built with a covenant for this purpose.

 Designed by Sir Edward Maufe and built between 1936 and 1961, it is the seat of the Diocese of Guildford.
</item 10>
=== END ITEM id=47aaac018c94d06d ===

=== ITEM id=9f474bdd7c822983 ===
Gold sentences and facts:
Sentence 0: Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.
  Fact 0.0: Big Fish is a musical.
  Fact 0.1: Big Fish has music by Andrew Lippa.
  Fact 0.2: Big Fish has lyrics by Andrew Lippa.
  Fact 0.3: Big Fish has a book by John August.
Sentence 1:  He is a resident artist at the Ars Nova Theater in New York City.
  Fact 1.0: He is a resident artist at the Ars Nova Theater.
  Fact 1.1: The Ars Nova Theater is in New York City.

Retrieved items:
<item 1>
Robert Lopez
Robert Lopez (born February 23, 1975) is an American songwriter of musicals, best known for co-creating "The Book of Mormon" and "Avenue Q", and for composing
the songs featured in the Disney animated film "Frozen".

 He is the youngest of only twelve people who have won an Emmy, a Grammy, an Oscar and a Tony Award, and the quickest (10 years) to win all four.
</item 1>

<item 2>
Marc Blitzstein
Blitzstein also composed music for films, such as "Surf and Seaweed" (1931) and "The Spanish Earth" (1937), and he contributed two songs to the original 1960
production of Hellman's play "Toys in the Attic".
</item 2>

<item 3>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 3>

<item 4>
Dan Regan
Dan Regan is a former trombone player for the Southern California-based ska punk band Reel Big Fish.

 He joined the band in 1994 and retired in October 2013.

 Regan has also played trombone in The Littlest Man Band with Scott Klopfenstein (formerly of Reel Big Fish) and currently has a hip-hop side project under the
alias Black Casper.

 He was the second longest lasting member of Reel Big Fish from 1994 until October 2013.

 He currently is starting a brewery in Long Beach, California.
</item 4>

<item 5>
Lin-Manuel Miranda, creator and star of *Hamilton* and *In the Heights*, wrote its music and lyrics and co-wrote songs for Disney’s *Moana*. His honors include
a Pulitzer, three Tonys, two Grammys, an Emmy, and a MacArthur Fellowship. The *Hamilton* cast album broke sales and chart records; its songs include “Your
Obedient Servant” and the Cabinet Battles. The text also covers Disney and Muppet soundtracks, *The M
</item 5>

<item 6>
Big Fish (musical)
Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.

 It is based on Daniel Wallace’s 1998 novel, "", and the 2003 film "Big Fish" written by August and directed by Tim Burton.
</item 6>

<item 7>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 7>

<item 8>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 8>

<item 9>
Andrew Lippa
Andrew Lippa (December 22, 1964) is an American composer, lyricist, book writer, performer, and producer.

 He is a resident artist at the Ars Nova Theater in New York City.
</item 9>

<item 10>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 10>
=== END ITEM id=9f474bdd7c822983 ===

=== ITEM id=f26bb05b1e9c74ba ===
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
Triple Canopy
Triple Canopy, Inc., is a private security company that provides integrated security, mission support and risk management services to corporate, government and
non-profit clients.

 The firm was founded in May 2003 by veteran U.S. Army Special Forces Soldiers, including former Delta Operators.

 In June 2014 the firm merged with rival security contracting firm, Academi, formerly Blackwater, thus forming the new company Constellis Group.

 The new CEO of Constellis Group is the former CEO of Academi, Craig Nixon, and training facilities are to be consolidated at the existing Academi training
facility in North Carolina.

 It was staffed by, among others, a number of former Army Special Operations personnel, Special Forces Soldiers, Rangers, SEALs, MARSOC Critical Skills
Operators, other special operations personnel, and a select few law enforcement officers.

 Over 5,000 employees worked for Triple Canopy at the time of the merger.
</item 1>

<item 2>
Dun &amp; Bradstreet
Dun & Bradstreet, Inc. () is a company that provides commercial data, analytics and insights for business, headquartered in Short Hills, a community in
Millburn, New Jersey, US.

 It offers a wide range of products, and services for risk and finance, operations and supply, and sales and marketing professionals, as well as research and
insights on global business issues, serving customers in government and industries such as communications, technology, strategic financial services, and
retail/telecommunications/manufacturing markets.

 Often referred to as D&B, the company’s database contains more than 265 million business records worldwide.
</item 2>

<item 3>
Joseph Saunders
Joseph W. Saunders (born c. 1945) is the executive chairman and former CEO of the multibillion-dollar global payments technology company Visa Inc., appointed in
2007.

 Before joining Visa International, he was assigned as president of card services for Washington Mutual, Inc. since acquiring Providian Financial Corporation in
October 2005.

 Saunders was president and CEO of Providian from November 2001, and chairman of the board from May 2002, until Washington Mutual’s Purchasing of Providian in
2005.

 From 1997 until 2001, Saunders served as chairman and CEO of Fleet Credit Card Services.
</item 3>

<item 4>
Mike Long (American businessman)
and returning it to profitability.

 Touching on his experience of bringing about the initial public offerings of web-based businesses, Long would say that investors needed to be presented "with
an entirely new face every few months," and that "the only way to run one of these Silicon Valley companies was to forget everything you'd learned outside of
Silicon Valley."
</item 4>

<item 5>
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
</item 5>

<item 6>
Summit Behavioral Healthcare
Summit Behavioral Healthcare (Summit BHC) is a Brentwood, Tennessee-based behavioral health services company that owns and operates a large network of addiction
treatment centers throughout the United States.

 Originating in 2012, the company was founded by Trey Carter, former CEO of Acadia Healthcare.

 In March 2015, Chicago, IL based investment firm Flexpoint Ford provided Round 1 funding to launch Summit BHC's growth plan.
</item 6>

<item 7>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 7>

<item 8>
Pramod Bhasin
Pramod Bhasin stepped down as President and CEO of Genpact, India's largest business process outsourcing (BPO) company, becoming non-executive Vice Chairman in
2011.
</item 8>

<item 9>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 9>

<item 10>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 10>
=== END ITEM id=f26bb05b1e9c74ba ===

=== ITEM id=5ac09dbf4b82371d ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.0: The Sports Museum is also known as The Sports Museum of New England.
  Fact 0.1: The Sports Museum is a non-profit museum.
  Fact 0.2: The Sports Museum is currently located in the TD Garden.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.
Sentence 1: TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.
  Fact 1.0: TD Garden is often called Boston Garden.
  Fact 1.1: TD Garden is often called "The Garden".
  Fact 1.2: TD Garden is a multi-purpose arena.
  Fact 1.3: TD Garden is in Boston, Massachusetts.
Sentence 2:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.0: TD Garden opened in 1995.
  Fact 2.1: TD Garden opened as a replacement for the original Boston Garden.
  Fact 2.2: TD Garden has been known as Shawmut Center.
  Fact 2.3: TD Garden has been known as FleetCenter.
  Fact 2.4: TD Garden has been known as TD Banknorth Garden.

Retrieved items:
<item 1>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 1>

<item 2>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 2>

<item 3>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 3>

<item 4>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 4>

<item 5>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 5>

<item 6>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 6>

<item 7>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 7>

<item 8>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 8>

<item 9>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 9>

<item 10>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 10>
=== END ITEM id=5ac09dbf4b82371d ===

=== ITEM id=a852246b088e59be ===
Gold sentences and facts:
Sentence 0: Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual art movement known as pop art.
  Fact 0.0: Andy Warhol was born Andrew Warhola.
  Fact 0.1: Andy Warhol's life span runs from August 6, 1928 to February 22, 1987.
  Fact 0.2: Andy Warhol was an American artist.
  Fact 0.3: Andy Warhol was an American director.
  Fact 0.4: Andy Warhol was an American producer.
  Fact 0.5: Andy Warhol was a leading figure in the visual art movement known as pop art.
Sentence 1: Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's production company.
  Fact 1.0: Alexander Butler was a British film director.
  Fact 1.1: Alexander Butler made over sixty features and short films.
  Fact 1.2: Alexander Butler made those features and short films during the 1910s and 1920s.
  Fact 1.3: Alexander Butler made many of those features and short films for G.B. Samuelson's production company.

Retrieved items:
<item 1>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 1>

<item 2>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 2>

<item 3>
Green Coca-Cola Bottles
With respect to its artistic merits, Warhol’s painting utilizes repetitive imagery in the shape of hundred and twelve almost identical bottles, evoking a sense
of mechanicalization which overlaps his former usage of mass culture objects.

 The widespread display of advertisement flooded the American public during the sixties, and Warhol had been successful in mapping it onto his canvas, no wonder
it is considered one of Warhol’s masterpieces.
</item 3>

<item 4>
Andy Warhol
Andy Warhol ( ; born Andrew Warhola; August 6, 1928 – February 22, 1987) was an American artist, director and producer who was a leading figure in the visual
art movement known as pop art.

 His works explore the relationship between artistic expression, celebrity culture, and advertising that flourished by the 1960s, and span a variety of media,
including painting, silkscreening, photography, film, and sculpture.

 Some of his best known works include the silkscreen paintings "Campbell's Soup Cans" (1962) and "Marilyn Diptych" (1962), the experimental film "Chelsea Girls"
(1966), and the multimedia events known as the "Exploding Plastic Inevitable" (1966–67).
</item 4>

<item 5>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 5>

<item 6>
John Warhola
John Warhola (May 31, 1925 – December 24, 2010) played a pivotal role in maintaining the legacy of his younger brother, pop artist Andy Warhol, assigned
responsibility by their father on his deathbed to ensure that Andy attended college and serving as a trustee of the Andy Warhol Foundation for the Visual Arts
after his brother's death in 1987.

 Warhola oversaw the establishment of The Andy Warhol Museum in Pittsburgh and the Andy Warhol Museum of Modern Art in Medzilaborce, Slovakia.
</item 6>

<item 7>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 7>

<item 8>
The Andy Warhol Story
The Andy Warhol Story (1966) is an underground film directed by Andy Warhol with cinematography by Paul Morrissey, and starring Edie Sedgwick and Rene Ricard
(as Andy Warhol).
</item 8>

<item 9>
David Fincher
David Andrew Leo Fincher (born August 28, 1962) is an American director and producer, notably for films, television series and music videos.

 He was nominated for the Academy Award for Best Director for "The Curious Case of Benjamin Button" (2008) and "The Social Network" (2010).

 For the latter, he won the Golden Globe Award for Best Director and the BAFTA Award for Best Direction.
</item 9>

<item 10>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 10>
=== END ITEM id=a852246b088e59be ===
