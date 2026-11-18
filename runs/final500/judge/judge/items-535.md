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

=== ITEM id=4e25261b3e17824b ===
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
List of Billboard 200 number-one albums of 2002
Country music singer Shania Twain's album "Up!

" opened up with a huge first week sales of 857,000 copies in its first week alone, giving her the recognition of the highest first week sales of her career and
second highest of the year, only behind Eminem's "The Eminem Show" and at the time the fastest selling solo female album ever.

 Nelly's album "Nellyville" opened up with his highest first week sales of his career which logged on with huge sales of 714,000 copies in its first week alone,
which beat his sales of his debut album "Country Grammar", which opened up with first week sales of 235,000 copies.

 Country singer Alan Jackson album "Drive" gave him his first number one album on the chart and opened up with first week sales of 211,000 copies alone.
</item 1>

<item 2>
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
MTV Video Music Award for Best New Artist
The MTV Video Music Award for Best New Artist have been given out since the first annual MTV Video Music Awards in 1984.

 From then to 2006, the award was named Best New Artist in a Video.

 In 2007 its name was changed to Best New Artist, as the category underwent a format change to award the artist's body of work for the full year rather than a
specific video.

 For the 2008 ceremony, though, while the award retained its 2007 name, it returned to the format of awarding a specific video rather than the artist's full
body of work.

 The category was later renamed Artist to Watch through 2013 to 2015 while still keeping the format of an award going to a certain video.
</item 4>

<item 5>
List of accolades received by Grand Theft Auto V
Within twenty-four hours of its release, "Grand Theft Auto V" generated more than $815.7 million in worldwide revenue, equating to approximately 11.21 million
copies sold for Take Two.

 Three days after release, the game had surpassed one billion dollars in sales, making it the fastest selling entertainment product in history.
</item 5>

<item 6>
Japan Record Award for Best New Artist
The Japan Record Award for Best New Artist (最優秀新人賞 ) is awarded annually.

 Until the 10th Japan Record Awards it was called New Artist Award, since 11th — Best New Artist Award.

 At present all nominees for the Best New Artist Award are awarded the New Artist Award.
</item 6>

<item 7>
List of Billboard 200 number-one albums of 2002
The highest-selling albums and EPs in the United States are ranked in the "Billboard" 200, published by "Billboard" magazine.

 The data are compiled by Nielsen Soundscan based on each album's weekly physical and digital sales.

 25 acts achieved number one albums during this year with artist such as Nelly and Shania Twain who had their albums debut at number one on the chart.

 Rapper Eminem's "The Eminem Show" is the best selling album of 2002 selling over approximately 7.6 million copies by the end of the year.

 It is also the longest running album of 2002 spending six non-consecutive weeks the chart and was known for its first full week of sales debut of 1.322 million
copies which Nielsen SoundScan scanned as the sixth largest sales of all time in its first week.

 Its debut of 1.322 million copies has still not been matched by any album today since except for Taylor Swift's album "1989", which opened with first week
sales of 1.279 million copies.
</item 7>

<item 8>
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
</item 8>

<item 9>
Robson &amp; Jerome (album)
Robson & Jerome is the debut studio album by British singing duo Robson & Jerome.

 It was released in the UK in 1995 by RCA Records, and reached number one on the UK Albums Chart.

 It was the Christmas number one album of 1995, and was the biggest-selling album of the year, with sales of 2,040,000.

 For almost 20 years, it held the record for selling 2 million copies in the UK in the fastest time, doing so in 42 days; in 2015, Adele surpassed the record
when her album "25" took just 29 days to reach the landmark.
</item 9>

<item 10>
Billboard Year-End
The same method was used for albums based on the "Billboard" 200, and songs appearing on the other charts (e.g. Hot Country Singles).
</item 10>
=== END ITEM id=4e25261b3e17824b ===

=== ITEM id=05a95a34801707c0 ===
Gold sentences and facts:
Sentence 0: Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.
  Fact 0.0: Jeffrey Charles "Jeff" Ragsdale is an American author.
  Fact 0.1: Jeffrey Charles "Jeff" Ragsdale is a documentary filmmaker.
  Fact 0.2: Jeffrey Charles "Jeff" Ragsdale is an actor.
  Fact 0.3: Jeffrey Charles "Jeff" Ragsdale is a stand-up comedian.
Sentence 1: Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.
  Fact 1.0: Millard Webb was born on 6 December 1893.
  Fact 1.1: Millard Webb died on 21 April 1935.
  Fact 1.2: Millard Webb was an American screenwriter.
  Fact 1.3: Millard Webb was an American director.

Retrieved items:
<item 1>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 1>

<item 2>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 2>

<item 3>
Graph query row:
person: Jeff Ragsdale
relationship: CREATED_BY
work: Jeff, One Lonely Guy
kind: book
</item 3>

<item 4>
Jeff Ragsdale
Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.

 In 2011 he posted a flyer in New York City as a "social experiment", stating his phone number and asking people to call him, describing himself as "Jeff, one
lonely guy".

 He was overwhelmed with thousands of calls after photos of the flyer were posted on the internet.

 The experience led to his 2012 book "Jeff, One Lonely Guy", and indirectly to a 2013 pilot episode for a reality television show, "Being Noticed", and a
starring role in the 2014 documentary "Hotline".
</item 4>

<item 5>
Julie Rivkin
Julie H. Rivkin (born 1952) is an American literary critic and professor of English at Connecticut College since 1982.

 She is best known for her publications on literary theory and Henry James, and has published several works on both subjects.

 Rivkin received her B.A. and PhD from Yale University and is currently the Associate Dean of Faculty at Connecticut College, a member of the Modern Language
Association, and Vice President of the Henry James Society.

 Her other specializations include American literature and gender studies (publisher of the "Henry James Review").
</item 5>

<item 6>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 6>

<item 7>
Ross Winans
Ross Winans (1796–1877) was an American inventor, mechanic, and builder of locomotives and railroad machinery.

 He is also noted for design of pioneering cigar-hulled ships.

 Winans, one of the United States' first multi-millionaires, was involved in national and state politics, a southern-sympathizer and was a vehement "states'
rights" advocate.

 His outspoken anti-federal stance as a member of the Maryland House of Delegates, the lower chamber of the General Assembly, (state legislature) led to his
temporary arrest on board a Baltimore and Ohio Railroad train returning from an early session of the legislature held in the western Maryland town of Frederick
to avoid the Union Army-occupied state capital of Annapolis in April–May, 1861, to consider the possibilities of state secession, during the early decisive
period of the American Civil War.

 Winans was related to James McNeill Whistler through marriage (Whistler's brother George married Winans' daughter Julia).
</item 7>

<item 8>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 8>

<item 9>
Jeff, One Lonely Guy
Jeff, One Lonely Guy is a 2012 nonfiction book by Jeff Ragsdale.

 It was published on March 20, 2012 by New Harvest.

 Dave Eggers selected the book for inclusion in The Best American Nonrequired Reading 2012, and it was a GQ 2012 "Book of the Year".

 In 2014 Amitava Kumar included portions of "Jeff, One Lonely Guy" in his newly released book, "A Matter of Rats: A Short Biography of Patna".

 Kumar previously interviewed Ragsdale and wrote about him in "The New York Times".
</item 9>

<item 10>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 10>
=== END ITEM id=05a95a34801707c0 ===

=== ITEM id=935c65a94a5b9d5e ===
Gold sentences and facts:
Sentence 0: Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's World Cup.
  Fact 0.0: Liu Ailing was born on June 2, 1967.
  Fact 0.1: Liu Ailing is a Chinese former footballer.
  Fact 0.2: Liu Ailing played for the China national team at the 1991 FIFA Women's World Cup.
  Fact 0.3: Liu Ailing played for the China national team at the 1995 FIFA Women's World Cup.
  Fact 0.4: Liu Ailing played for the China national team at the 1999 FIFA Women's World Cup.
Sentence 1: The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale de Football Association" (FIFA), the sport's global governing body.
  Fact 1.0: The FIFA Women's World Cup is an international football competition.
  Fact 1.1: The FIFA Women's World Cup is contested by the senior women's national teams of the members of FIFA.
  Fact 1.2: FIFA stands for Fédération Internationale de Football Association.
  Fact 1.3: FIFA is the sport's global governing body.

Retrieved items:
<item 1>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 1>

<item 2>
Geri Donnelly
Geraldine "Geri" Donnelly (born 30 November 1965) is a Canadian former soccer player.

 A midfielder, she represented Canada at the 1995 and 1999 editions of the FIFA Women's World Cup and was named Canadian Player of the Year in 1996 and 1999.

 Donnelly was part of the Canadian squad who won the 1998 CONCACAF Women's Championship.

 She was selected as a member of the Canadian Soccer Hall of Fame in 2014.
</item 2>

<item 3>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 3>

<item 4>
Liu Ailing
Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's
World Cup.

 She won a silver medal at the 1996 Atlanta Olympics and participated at the 2000 Sydney Olympics.

 A playmaking midfielder, she played professional club football in Japan and the United States.
</item 4>

<item 5>
EAFF E-1 Football Championship (women)
EAFF E-1 Football Championship is an international football competition in East Asia for national teams of the East Asian Football Federation (EAFF).

 The competition between women's national teams is held alongside men's competition.
</item 5>

<item 6>
FIFA Women's World Cup
The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale
de Football Association" (FIFA), the sport's global governing body.

 The competition has been held every four years since 1991, when the inaugural tournament, then called the Women's World Championship, was held in China.
</item 6>

<item 7>
Women's Nordic Football Championship
Women's Nordic Football Championship was an international football competition contested by the women's national football teams of the Nordic countries.

 The tournament was held annually between 1974 and 1982.

 Finland, Denmark and Sweden competed from the start, Norway joined the tournament in 1978.

 Iceland and Faroe Islands did not take part at the competition.
</item 7>

<item 8>
Gambia women's national football team
The Gambia women's national football team represents the Gambia in international football competition.

 The team, however, has not competed in a match recognised by FIFA, the sport's international governing body, despite that organised women's football has been
played in the country since 1998.

 The Gambia has two youth teams, an under-17 side that has competed in FIFA U-17 Women's World Cup qualifiers, and an under-19 side that withdrew from regional
qualifiers for an under-19 World Cup.

 The development of a national team faces challenges similar to those across Africa, although the national football association has four staff members focusing
on women's football.
</item 8>

<item 9>
EAFF E-1 Football Championship
EAFF E-1 Football Championship, known as the East Asian Football Championship from 2003 to 2010, and the EAFF East Asian Cup for the 2013 and 2015 editions, is
a men's international football competition in East Asia for member nations of the East Asian Football Federation (EAFF).

 Before the EAFF was founded in 2002, the Dynasty Cup was held between the East Asian top four teams, and was regarded as the unofficial East Asian
Championship.

 There is a separate competition for both men (first held in 2003) and women (first held in 2005).

 There was also a combined points competition in 2005, where the results of the men's and women's teams are added together (not including qualifiers).
</item 9>

<item 10>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 10>
=== END ITEM id=935c65a94a5b9d5e ===

=== ITEM id=c1682adc57a12b75 ===
Gold sentences and facts:
Sentence 0: Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
  Fact 0.0: Leitneria floridana is also called corkwood.
  Fact 0.1: Leitneria floridana is the sole species in the genus Leitneria.
  Fact 0.2: Leitneria floridana is a deciduous dioecious shrub or small tree.
  Fact 0.3: Leitneria floridana is found only in the southeastern United States states of Arkansas, Florida, Georgia, Missouri and Texas.
Sentence 1: Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.
  Fact 1.0: Tricyrtis is a genus of Asian flowering plants.
  Fact 1.1: Tricyrtis is in the lily family.
  Fact 1.2: Tricyrtis has approximately 20 known species.

Retrieved items:
<item 1>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 1>

<item 2>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 2>

<item 3>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 3>

<item 4>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 4>

<item 5>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 5>

<item 6>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 6>

<item 7>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 7>

<item 8>
Tricyrtis
Tricyrtis is a genus of Asian flowering plants in the lily family, with approximately 20 known species.

 The species are commonly known in English as toad lilies.

 The genus has a native range from the Himalayas to eastern Asia, including China, Japan, Philippines and Taiwan, and a few species are cultivated for their
ornamental qualities in other parts of the world.
</item 8>

<item 9>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 9>

<item 10>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 10>
=== END ITEM id=c1682adc57a12b75 ===

=== ITEM id=e9c486e8357c29b7 ===
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
OK Go videography
The success of OK Go's music first won the band the 14th Annual Webby Special Achievement Award for Film and Video Artist of the Year.

 The video for "This Too Shall Pass" was named both "Video of the Year" and "Best Rock Video" at the 3rd annual UK Music Video Awards.

"This Too Shall Pass" won the LA Film Fest's Audience Award for Best Music Video, UK MVA Awards – Music Video of the Year Winner 2010, among others.
</item 1>

<item 2>
Twelve Days of OK Go
Twelve Days of OK Go is a compilation album by American rock band OK Go.

 It was released on December 31, 2012.

 OK Go started releasing the songs on December 10, with one song released each weekday.

 The last song, a cover of "Any Time at All", was released on Christmas.

 A bonus track, a cover of "This Will Be Our Year," was released on New Year's Eve.
</item 2>

<item 3>
OK Go (2000 EP)
OK Go or The Brown EP (officially known as OKGoCD.001) is the first EP by American rock band OK Go.
</item 3>

<item 4>
Graph query row:
entity: OK Go
labels: ['__KGBuilder__', '__Entity__', 'Organization']
genre: rock
classification: band
linked_genre: None
</item 4>

<item 5>
OK Go (album)
OK Go is the debut studio album by American rock band OK Go.

 It was released in September 2002.

 The cover was created by designer Stefan Sagmeister.
</item 5>

<item 6>
Distortion Mirrors
Distortion Mirrors is an English alternative rock band, based in Leeds, West Yorkshire, England, and who was formed in 2011.
</item 6>

<item 7>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 7>

<item 8>
The Writing's on the Wall (OK Go song)
"The Writing's on the Wall" is a song by American rock band OK Go.

 It was released on June 17, 2014, as part of the band's EP "Upside Out", and is also the first single from the band's fourth studio album "Hungry Ghosts".

 On the same day, the band released a music video in which the members use props to create optical illusions, reflecting the song's description of a
relationship that fails because the couple has different points of view.

 Like previous OK Go videos, it is structured as a one-shot music video.

 The many YouTube views of the video caused the song to debut in the top ten of the US "Billboard" Hot Rock Songs chart, as well as number one on the Bubbling
Under Hot 100 Singles chart.
</item 8>

<item 9>
OK Go (2001 EP)
OK Go or The Pink EP (officially known as OKGoCD.002) is the second EP by American rock band OK Go.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=e9c486e8357c29b7 ===

=== ITEM id=10a50ab9edd25b9f ===
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
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 1>

<item 2>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 2>

<item 3>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 3>

<item 4>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 4>

<item 5>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 5>

<item 6>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 6>

<item 7>
Carlina
Carlina is a genus of flowering plants in the aster family, Asteraceae.

 It is distributed from Madeira and the Canary Islands across Europe and northern Africa to Siberia and northwestern China.
</item 7>

<item 8>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 8>

<item 9>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 9>

<item 10>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 10>
=== END ITEM id=10a50ab9edd25b9f ===

=== ITEM id=8a0e2d595d9c0d9e ===
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
Light Surrounding You
"Light Surrounding You" is the second single by alternative rock band, Evermore, taken from their second studio album, "Real Life" (June 2006).

 It was written by Dann Hume, the group's drummer and lead singer and guitarists Jon Hume.

 The record is co-produced by Jon Hume.

 The song was released in October 2006 and peaked at No. 15 on the RIANZ Singles Chart in New Zealand and number one on the ARIA Singles Chart, making it
Evermore's most successful single in Australia.

 It was the first single by a New Zealand artist to top the Australian charts since "How Bizarre" by OMC in 1996.
</item 3>

<item 4>
He Will Have His Way
He Will Have His Way is a compilation album featuring male Australian and New Zealand musicians performing songs written by Neil Finn and Tim Finn (The Finn
Brothers), who are best known as members of Split Enz and Crowded House.

 It was released on 12 November 2010 and is a sequel to "She Will Have Her Way", a 2005 album featuring female Australian and New Zealand musicians.

 The two albums were later released together with extra tracks as the double CD "They Will Have Their Way".
</item 4>

<item 5>
Jon Hume
Jon Hume (born 6 November 1983) is the lead singer of Evermore.

 He is a singer-songwriter, multi-instrumentalist, Record Producer and Studio owner.

 Jon is the eldest of the three brothers who make up the New Zealand band Evermore.
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
Nevermore
Nevermore is an American progressive metal band from Seattle, Washington, United States.

 Formed in 1991, they are known to incorporate elements from various styles such as power metal, modern hard rock, classic heavy metal and technical thrash
metal.

 The band has been on hiatus since 2011, due to personal issues between the band members.

 In April 2015, lead singer Warrel Dane confirmed that Nevermore has not disbanded, and there is a possibility for them to continue in the next two years with
another album.
</item 7>

<item 8>
Between the Lines (Evermore song)
"Between the Lines" is the first single by Evermore, taken from their third studio album "".

 Evermore's Jon Hume said that "Between the Lines was the first song to come out of a search for a new musical experience as a band."

 It was released as a free download on Evermore's official website on 10 November 2008.
</item 8>

<item 9>
Evermore (Evermore album)
Evermore is the self-titled international debut album by New Zealand rock band Evermore, which was also released in Australia and New Zealand on 12 March 2010.

 The album is a collection of songs from the band's entire history, as well as three new songs including the single "Underground".
</item 9>

<item 10>
Running (Evermore song)
"Running" is the lead single by alternative rock band Evermore, taken from their second studio album, "Real Life" (July 2006).

 It was written by backing vocalist and drummer, Dann Hume.

 It was issued on 3 June, ahead of the album.

 It was co-produced by Jon with John Alagía (Dave Matthews Band, John Mayer).

 It peaked at No. 4 on their native RIANZ Singles Chart in New Zealand and No. 5 on their adopted country of Australia's ARIA Singles Chart.
</item 10>
=== END ITEM id=8a0e2d595d9c0d9e ===

=== ITEM id=5b1aa7ef468945ee ===
Gold sentences and facts:
Sentence 0: The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
  Fact 0.0: The Second Battle of Guam was fought from 21 July to 10 August 1944.
  Fact 0.1: The Second Battle of Guam was the American recapture of the Japanese-held island of Guam.
  Fact 0.2: Guam is a U.S. territory in the Mariana Islands.
  Fact 0.3: Japan captured Guam from the U.S. in the 1941 First Battle of Guam.
  Fact 0.4: The Second Battle of Guam took place during the Pacific campaign of World War II.
Sentence 1: The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and Antietam Creek as part of the Maryland Campaign.
  Fact 1.0: The Battle of Antietam was also known as the Battle of Sharpsburg, particularly in the South.
  Fact 1.1: The Battle of Antietam was fought on September 17, 1862.
  Fact 1.2: The Battle of Antietam was fought near Sharpsburg, Maryland.
  Fact 1.3: The Battle of Antietam was fought near Antietam Creek.
  Fact 1.4: The Battle of Antietam was part of the Maryland Campaign.

Retrieved items:
<item 1>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
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
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
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
Battle of Antietam
The Battle of Antietam , also known as the Battle of Sharpsburg, particularly in the South, was fought on September 17, 1862, near Sharpsburg, Maryland and
Antietam Creek as part of the Maryland Campaign.

 It was the first field army–level engagement in the Eastern Theater of the American Civil War to take place on Union soil and is the bloodiest single-day
battle in American history, with a combined tally of 22,717 dead, wounded, or missing.
</item 5>

<item 6>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 6>

<item 7>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 7>

<item 8>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 8>

<item 9>
Maryland Campaign
The Maryland Campaign—or Antietam Campaign—occurred September 4–20, 1862, during the American Civil War.

 Confederate Gen. Robert E. Lee's first invasion of the North was repulsed by the Army of the Potomac under Maj. Gen. George B. McClellan, who moved to
intercept Lee and his Army of Northern Virginia and eventually attacked it near Sharpsburg, Maryland.

 The resulting Battle of Antietam was the bloodiest single-day battle in American history and is widely considered one of the major turning points of the war.
</item 9>

<item 10>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 10>
=== END ITEM id=5b1aa7ef468945ee ===

=== ITEM id=42269e706e6de9cd ===
Gold sentences and facts:
Sentence 0:  The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
  Fact 0.0: The first of the books was optioned by Universal Pictures.
  Fact 0.1: The option followed a bidding war over the manuscript.
Sentence 1:  The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film, and the oldest in terms of the overall film market .
  Fact 1.0: The company was founded in 1912.
  Fact 1.1: The company was founded by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour.
  Fact 1.2: The company is the oldest surviving film studio in the United States.
  Fact 1.3: The company is the world's fourth oldest film studio, after Gaumont, Pathé and Nordisk Film.
  Fact 1.4: The company is the oldest film studio in terms of the overall film market.

Retrieved items:
<item 1>
Graph query row:
organization: Universal Pictures
founded_on: 1912
person: None
connection: None
</item 1>

<item 2>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 2>

<item 3>
Applewood Books
Applewood Books is a book publishing company founded by Phil Zuckerman in 1976.

 They specialize in publishing exacting recreations of historic books, including complex reprints of children's art and pop-up books and other books published
by methods which duplicate antique publishing techniques.

 They have more than 2000 titles in print.

 In recent years, the company has been working to increase the number of reprints it publishes.

 In 2007, the company published over 300 titles.
</item 3>

<item 4>
Villard (imprint)
Villard, also known as Villard Books, is a publishing imprint of Random House, one of the largest publishing companies in the world.

 It was founded in 1983.

 Villard began as an independent imprint of Random House and is currently a sub-imprint of Ballantine Books, itself an imprint of Random House.

 It was named after a Stanford White brownstone mansion on Madison Avenue that was the home of Random House for twenty years.
</item 4>

<item 5>
Field Enterprises
Field Enterprises was a private holding company founded on August 31, 1944, by Marshall Field III and others whose main asset was the "Chicago Sun".

 That same year the company acquired the book publishers Simon & Schuster and Pocket Books.
</item 5>

<item 6>
Bogle-L'Ouverture Publications
Bogle-L’Ouverture Publications (BLP) is a radical London-based publishing company founded in 1968 by Guyanese activists Jessica Huntley (23 February 1927 – 13
October 2013) and Eric Huntley (born 25 September 1929).

 Named in honour of two outstanding liberation fighters in Caribbean history, Toussaint L'Ouverture and Paul Bogle, the company began operating during a period
in the UK when "books by Black authors or written with a sympathetic view of Black people’s history and culture were rare in mainstream bookshops in the UK."

 Alongside New Beacon Books (founded in 1966) and Allison & Busby (founded in 1967), BLP was one of the first black-owned independent publishing companies in
the UK.

 BLP has been described as "a small, unorthodox, self-financing venture that brought a radical perspective to non-fiction, fiction, poetry and children's
books."
</item 6>

<item 7>
Simon Bloom
Simon Bloom is a series of young adult novels by Michael Reisman.

 Set in the fictional town of Lawnville, New Jersey, it is about a boy who finds a book that lets him control the laws of physics.

 The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
</item 7>

<item 8>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 8>

<item 9>
Mimeo, Inc
Mimeo.com, Inc. is a privately held Print on demand and digital distribution document company.

 It was the first to offer online printing and overnight delivery of complex documents and marketing materials.

 The company refers to itself as a technology company that prints.

 Customers utilize a proprietary online workflow connected to multiple print production, warehouse and distribution centers.

 Customers include small, mid-sized and large companies.

 The company was named after the Mimeograph.

 Printing and distribution centers are located in Memphis, Tennessee, Newark, New Jersey and Huntingdon, Cambridgeshire, England.
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=42269e706e6de9cd ===

=== ITEM id=47c67bffbdcd5444 ===
Gold sentences and facts:
Sentence 0: Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France (1933–1940 and 1950–1957), and the United States (1947–1950).
  Fact 0.0: Maximillian Oppenheimer was born on 6 May 1902.
  Fact 0.1: Maximillian Oppenheimer died on 26 March 1957.
  Fact 0.2: Maximillian Oppenheimer was known as Max Ophüls.
  Fact 0.3: Max Ophüls was a German-born film director.
  Fact 0.4: Max Ophüls worked in Germany from 1931 to 1933.
  Fact 0.5: Max Ophüls worked in France from 1933 to 1940.
  Fact 0.6: Max Ophüls worked in France from 1950 to 1957.
  Fact 0.7: Max Ophüls worked in the United States from 1947 to 1950.
Sentence 1: Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.
  Fact 1.0: Shirley Clarke was born on October 2, 1919.
  Fact 1.1: Shirley Clarke died on September 23, 1997.
  Fact 1.2: Shirley Clarke was an American experimental filmmaker.
  Fact 1.3: Shirley Clarke was an American independent filmmaker.

Retrieved items:
<item 1>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 1>

<item 2>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 2>

<item 3>
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
</item 3>

<item 4>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 4>

<item 5>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 5>

<item 6>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 6>

<item 7>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 7>

<item 8>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 8>

<item 9>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 9>

<item 10>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 10>
=== END ITEM id=47c67bffbdcd5444 ===

=== ITEM id=a2296eaf48027feb ===
Gold sentences and facts:
Sentence 0:  Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
  Fact 0.0: Programming is simulcast on WANH (91.5 FM).
  Fact 0.1: WANH (91.5 FM) is located in Meredith.
  Fact 0.2: WANH serves the Lakes Region.
Sentence 1:  The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford County and Merrimack County.
  Fact 1.0: The area comprises all of Belknap County.
  Fact 1.1: The area comprises the southern portion of Carroll County.
  Fact 1.2: The area comprises the eastern portion of Grafton County.
  Fact 1.3: The area comprises the northern portion of Strafford County.
  Fact 1.4: The area comprises the northern portion of Merrimack County.

Retrieved items:
<item 1>
WLMR
WLMR (1450 AM) is a radio station broadcasting a religious format.

 Licensed to Chattanooga, Tennessee, USA, the station serves the Chattanooga area.

 The station is currently owned by Wilkins Communications Network, Inc. and features programming from USA Radio Network.

 In the early 1980s, the station was automated and played country music from studios in a strip mall on Brainerd Rd.

 It was also Chattanooga's first talk radio station when it held the WZRA call letters.

 WZRA was the first home of Jeff Styles, and also featured well known personalities such as Kelly McCoy and Robert T. Nash.
</item 1>

<item 2>
WVNH
WVNH (91.1 FM) is a radio station broadcasting a Christian radio format.

 Licensed to Concord, New Hampshire, United States, the station serves the Concord area.

 The station is currently owned by New Hampshire Gospel Radio, Inc.

 Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
</item 2>

<item 3>
KMBC-TV
KMBC-TV, virtual channel 9 (UHF digital channel 29), is the ABC-affiliated television station licensed to Kansas City, Missouri, United States and also serving
Kansas City, Kansas.

 The station is owned by the Hearst Television subsidiary of the Hearst Corporation, as part of a duopoly with CW affiliate KCWE (channel 29).

 The two stations share studio facilities located at the Winchester Business Center on Winchester Avenue and East 63rd Street (near Swope Park, off of I-435) in
southeastern Kansas City, Missouri; KMBC-TV maintains transmitter facilities located near the Blue River in eastern Kansas City.

 On cable, KMBC-TV is available on Charter Spectrum, Comcast Xfinity and Consolidated Communications channel 12, and Google Fiber and AT&T U-verse channel 9.

 There is a high definition feed provided on Spectrum digital channel 1200, Xfinity channel 812, Consolidated channel 610 and U-verse channel 1009.
</item 3>

<item 4>
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 4>

<item 5>
KVKI-FM
KVKI-FM (96.5 FM) is a radio station broadcasting an adult contemporary music format.

 Licensed to Shreveport, Louisiana, United States, the station serves the Shreveport area.

 The station is currently owned by Townsquare Media.

 Its studios are shared with its other five sister stations in West Shreveport (one mile west of Shreveport Regional Airport), and the transmitter is in
Blanchard, Louisiana.

 The call letters were originally KBCL, and later became religious station KEPT, before the call letters were changed to KVKI to be similar to the call letters
of the popular KVIL radio station in Dallas, Texas.
</item 5>

<item 6>
KSCW-DT
KSCW-DT, virtual channel 33 (VHF digital channel 12), is a CW-affiliated television station licensed to Wichita, Kansas, United States.

 The station is owned by Gray Television, as part of a duopoly with CBS affiliate KWCH-DT (channel 12); Gray also operates Univision affiliate KDCU-DT (channel
31) under a joint sales agreement with owner Entravision Communications Corporation.

 KSCW and KWCH share studio facilities and KDCU's master control operations are located on East 37th Street North in northeastern Wichita; KSCW maintains
transmitter facilities located in rural northeastern Reno County (due south of Buhler).
</item 6>

<item 7>
KKBN
KKBN (93.5 FM) The Cabin a radio station since 1986 broadcasting a Country music format since March 2000.

 Licensed to Twain Harte, California, United States, the station serves the greater Mother Lode area comprising Tuolumne and Calaveras Counties with a combined
population of approximately 110,000.

 The station is currently owned by Clarke Broadcasting Corporation.
</item 7>

<item 8>
KRBT
KRBT (1340 AM) is a radio station broadcasting a sports format.

 Licensed to Eveleth, Minnesota, United States, the station serves the Iron Range area.

 The station is currently owned by Range Broadcasting, Inc., and features programming from the Red Zone Radio Network.

 Originally a stand alone music station with call letters WEVE, then a simulcast music station with WEVE-FM, the station changed it's call letters to KRBT for
"Range's Best Talk" in June of 1998 and discontinued simulcast with WEVE-FM to become a talk station.

 The format change to sports/talk was made following the death of previous owner, Lew Latto, Iron Range Broadcasting, Inc.

 President on August, 24 2011 and subsequent sale to Red Rock Radio in March 2012, and has been retained by the new owner as of July 2017, Range Broadcasting,
Inc.
</item 8>

<item 9>
KWXL-LP
KWXL-LP (98.7 FM) is a high school radio station broadcasting a variety format.

 Licensed to Tucson, Arizona, United States, the station serves the Tucson area.

 The station is currently owned by the Tucson Unified School District.

 KWXL-LP is Tucson's only high school radio station.

 It is also a news radio station for students who attend Pueblo High Magnet School.

 Students broadcasting over the radio station are enrolled in a "Writing/Reporting for Broadcasting" class with instructor Sarah Walson.

 Originally started by Douglas Potter, who retired in 2006.

 Listeners can also find FM 98.7, KWXL on iTunes.
</item 9>

<item 10>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 10>
=== END ITEM id=a2296eaf48027feb ===

=== ITEM id=f543b8238d88cdce ===
Gold sentences and facts:
Sentence 0: Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".
  Fact 0.0: Stephanocereus is a genus of cactus.
  Fact 0.1: Stephanocereus is from Brazil.
  Fact 0.2: Stephanocereus is related to "Arrojadoa".
Sentence 1:  They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).
  Fact 1.0: The plants in this sentence are mostly liana-forming.
  Fact 1.1: The plants in this sentence are found in the Old World tropics.
  Fact 1.2: The plants in this sentence range from South China, Indonesia, Malaysia and the Philippines.
  Fact 1.3: The plants in this sentence range westward to Madagascar, with two species there.
  Fact 1.4: The plants in this sentence range westward to the Seychelles, with one species there.
  Fact 1.5: The plants in this sentence range southward to Australia, with three species there.
  Fact 1.6: The plants in this sentence range southward to New Caledonia, with one species there.
  Fact 1.7: The plants in this sentence range northward to India, with one species there.
  Fact 1.8: The plants in this sentence range northward to Sri Lanka, with one species there.

Retrieved items:
<item 1>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 1>

<item 2>
Nepenthes tenax
Nepenthes tenax ( ; from Latin: "tenax" "tenacious") is a lowland species of tropical pitcher plant native to northern Queensland, Australia.

 It is the third "Nepenthes" species recorded from the continent and its second endemic species.

 "Nepenthes tenax" is closely related to the two other Australian "Nepenthes" species: "N. mirabilis" and "N. rowaniae".
</item 2>

<item 3>
Nepenthes chaniana
Nepenthes chaniana ( ; after Datuk Chan Chew Lun, Managing Director of Natural History Publications (Borneo)) is a tropical pitcher plant species belonging to
the genus "Nepenthes".

 It is characterised by a dense indumentum of long, white hairs.

 Pitchers are cylindrical and mostly white to yellow in colouration.

 "Nepenthes chaniana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. boschiana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 3>

<item 4>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 4>

<item 5>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 5>

<item 6>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 6>

<item 7>
Nepenthes × kinabaluensis
Nepenthes × kinabaluensis , or the Kinabalu Pitcher-Plant, is the natural hybrid between "N. rajah" and "N. villosa".

 It was first collected near Kambarangoh on Mount Kinabalu, Borneo by Lilian Gibbs in 1910 and later mentioned by John Muirhead Macfarlane as ""Nepenthes" sp."

 in 1914.

 Although Macfarlane did not formally name the plant, he noted that "[a]ll available morphological details suggest that this is a hybrid between "N. villosa"
and "N. rajah"".

 It was finally described in 1976 by Shigeo Kurata as "N. × kinabaluensis".

 The name was first published in "Nepenthes of Mount Kinabalu", but was a "nomen nudum" at the time as it lacked an adequate description and information on the
type specimen.

 The name was subsequently published validly by Kurata in 1984.
</item 7>

<item 8>
Robert Cantley
Robert Cantley is a conservationist and Managing Director of Borneo Exotics, a Sri Lankan-based plant nursery specialising in tissue-cultured and seed-grown
"Nepenthes" species and hybrids.

 Cantley has contributed to a number of papers on "Nepenthes".

 The species "Nepenthes robcantleyi", the natural hybrid "Nepenthes × cantleyi" and the cultivar "Nepenthes" 'Cantley's Red' are named after him.
</item 8>

<item 9>
Nepenthes boschiana
Nepenthes boschiana ( ; after Johannes van den Bosch), or Bosch's Pitcher-Plant, is a tropical pitcher plant endemic to Borneo.

 It is most closely allied to "N. faizaliana".

 "Nepenthes borneensis" is considered a synonym of this species.

 "Nepenthes boschiana" has no known natural hybrids.

 No valid forms or varieties have been described.

 "Nepenthes boschiana" belongs to the loosely defined ""N. maxima" complex", which also includes, among other species, "N. chaniana", "N. epiphytica",
"N. eymae", "N. faizaliana", "N. fusca", "N. klossii", "N. maxima", "N. platychila", "N. stenophylla", and "N. vogelii".
</item 9>

<item 10>
Nepenthes veitchii
Nepenthes veitchii ( or ; after James Veitch, nurseryman of the Veitch Nurseries), or Veitch's Pitcher-Plant, is a "Nepenthes" species from the island of
Borneo.

 The plant is widespread in north-western Borneo and can also be found in parts of Kalimantan.

 "Nepenthes veitchii" usually grows as an epiphyte, though the form from Bario seems to be strictly terrestrial and has not been observed to climb trees.
</item 10>
=== END ITEM id=f543b8238d88cdce ===
