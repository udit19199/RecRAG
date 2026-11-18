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

This file has 13 items.

=== ITEM id=df8e612d5a41a49f ===
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
Kinetic Records
The label was distributed by Reprise/Warner Bros.

 Records from its inception until 2001, when Kinetic switched distribution to BMG.

 The latter eventually absorbed the label during the merger with Sony Music.
</item 1>

<item 2>
Interscope Records
Interscope Records is an American major record label.

 An imprint of Interscope Geffen A&M Records, its parent company is Universal Music Group, a subsidiary of Vivendi S.A.
</item 2>

<item 3>
Kinetic Records
Kinetic Records was a New York City-based record label founded by Steve Lau (founding member of Sire Records act the Ocean Blue), that launched the careers of
artists and DJs such as Paul Oakenfold, BT, Ray Munns, Deepsky, Kosheen, Tilt, Timo Maas, Grace, Sandra Collins, Billy Thermal, Binary Finary, and Shpongle.

 Other successful artists on the label included Ash, Sasha and John Digweed, South, Faithless, DJ DAN, LTJ Bukem, Dario G, Luomo, Infected Mushroom, Max Graham,
Dave Ralph Music Instructor, and Hybrid.

 It was also well known for its Tranceport series, of which Paul Oakenfold's first installment is considered to be the breakthrough record for the trance genre
with the highest sales of any mix compilation.

 The release of Sasha's Airdrawndagger and Sasha and Digweed's mix compilation "Communicate" were also big releases for the label.

 Kinetic was the exclusive distributor of the Perfecto label in the US as well as the Another Late Night compilation series, which featured Howie B., Fila
Brazillia, and Rae and Christian.
</item 3>

<item 4>
The Foundation (2008), Zac Brown Band’s major-label debut, sold 3.4 million US copies by 2015, earned Grammy and ACM nominations, and won the band a Best New
Artist Grammy. Tevin Campbell’s Babyface-produced “Can We Talk” reached Hot 100 #9, topped the R&B chart for three weeks, went gold, and won a Soul Train award.
John Cena’s 2005 album introduced his entrance theme; Bow Wow’s “Fresh Azim
</item 4>

<item 5>
Love Incredible
"Love Incredible" is a song by Norwegian DJ and record producer Cashmere Cat, featuring vocals by Cuban-American singer Camila Cabello.

 It was released on 17 February 2017 through Interscope and Mad Love as the third single from his debut album, "9".
</item 5>

<item 6>
Paracadute
Paracadute is a record company owned by OK Go.

 OK Go left their previous record company, EMI, because of a disagreement between the band and the label.

 OK Go reissued their album, Of the Blue Colour of the Sky, with Paracadute taking care of the promotion and distribution of the album reissues.

 OK Go's lead singer, Damian Kulash, announced the new label in a video posted to the band's YouTube channel.
</item 6>

<item 7>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 7>

<item 8>
Incredible Connection
Incredible Connection (Pty) Ltd is the largest consumer electronics and IT retailer in South Africa.

 On 15 December 1998, Incredible Connection was acquired by Connection Group Holdings Limited and now operates as subsidiary of that company, itself a
subsidiary of JD Group Limited.

 As of 2005, Incredible Connection has 34 retail outlets in South Africa, and in 2002 it was reported to hold 45% of the IT retail market in the country.
</item 8>

<item 9>
Recess Records
Recess Records is an independent record label founded in 1989 with the release of F.Y.P's "Extra Credit", an 11 song 7-inch record.

 Label founder Todd Congelliere (frontman for F.Y.P and Toys That Kill) originally created the label as a vehicle to release F.Y.P's material.

 Congelliere promoted "Extra Credit" by running classified ads in Berkeley based Maximum Rocknroll fanzine and going to skateboard contests.

 He quickly sold his initial pressing of 500 copies.

 In between touring the U.S., Europe, and Japan, Congelliere began releasing records for fellow punk rock and DIY bands, eventually having a roster of close to
20 artists on his label.

 Pinhead Gunpowder, fronted by Green Day frontman Billie Joe Armstrong, announced in April 2008 that a new 7" single from the band would be released on Recess
Records in late May 2008.
</item 9>

<item 10>
American Record Corporation
American Record Corporation (ARC), also referred to as American Record Company, American Recording Corporation, or (erroneously) as ARC Records, was an American
record company.

 It resulted from the merger in 1929 of three companies: the Cameo Record Corporation (which owned Cameo, Lincoln and Romeo Records), the Pathé Phonograph and
Radio Corporation (which owned Actuelle, Pathé, and Perfect), and the Plaza Music Company (which owned Banner, Domino, Jewel, Oriole, and Regal).
</item 10>
=== END ITEM id=df8e612d5a41a49f ===

=== ITEM id=35d6ba84f0876b91 ===
Gold sentences and facts:
Sentence 0:  He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth Games in Edinburgh, Scotland.
  Fact 0.0: He also participated in the 10,000 metres at the 1967 Pan American Games.
  Fact 0.1: The 1967 Pan American Games were held in Winnipeg, Manitoba, Canada.
  Fact 0.2: He represented Guyana at the 1970 British Commonwealth Games.
  Fact 0.3: The 1970 British Commonwealth Games were held in Edinburgh, Scotland.
Sentence 1:  The race consists of 25 laps around an Olympic-sized track.
  Fact 1.0: The race consists of 25 laps.
  Fact 1.1: The 25 laps are around an Olympic-sized track.

Retrieved items:
<item 1>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 1>

<item 2>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 2>

<item 3>
Dave Bailey (athlete)
David Bailey (born March 17, 1945 in Toronto, Ontario) is a retired track and field athlete, who represented Canada at the 1968 Summer Olympics in the men's
1.500 metres.

 He was the first Canadian to run the mile in less than 4 minutes (3:59.1) in San Diego, CA on June 11, 1966 and the first Canadian to the run the mile in less
than 4 minutes in Canada (3:57.7)in Toronto on July 22, 1967.

 A resident of Willowdale, Ontario he won the bronze medal in this event at the 1967 Pan American Games and the silver medal in this event at the 1968 World
University Games He was two-time Canadian Universities Track and Field Athlete of Year (1965, 1967) and two-time inductee into the University of Toronto Sports
Hall of Fame (individually 1998 and team 2003).
</item 3>

<item 4>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 4>

<item 5>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 5>

<item 6>
Nora Rocha
Nora Leticia Rocha de la Cruz (born December 18, 1967 in Monclova, Coahuila) is a retired female track and field athlete from Mexico, who competed in the 5000
and 10,000 metres.

 She claimed the gold medal in the women's 10,000 metres at the 1999 Pan American Games in Winnipeg, Canada.
</item 6>

<item 7>
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 7>

<item 8>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 8>

<item 9>
Ross MacKenzie (athlete)
Ross MacKenzie (born 18 July 1946) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1968 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Brian MacLaren, Bill Crothers, and Robert McLaren).

 MacKenzie also finished fifth in the 1967 Pan American Games 400 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Brian MacLaren and Bill Crothers.
</item 9>

<item 10>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 10>
=== END ITEM id=35d6ba84f0876b91 ===

=== ITEM id=3f3254ac2b81494a ===
Gold sentences and facts:
Sentence 0: And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
  Fact 0.0: And There's More is a comedy sketch show.
  Fact 0.1: And There's More stars Jimmy Cricket.
  Fact 0.2: And There's More was produced by Central Independent Television.
  Fact 0.3: And There's More was produced for ITV.
  Fact 0.4: And There's More was produced from 1985 till 1988.
Sentence 1:  He currently lives with his family in Rochdale, North West England.
  Fact 1.0: The person described in this passage currently lives with his family in Rochdale, North West England.
  Fact 1.1: The person described in this passage currently lives in Rochdale.
  Fact 1.2: Rochdale is in North West England.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 2>

<item 3>
Zachary Throne
As a singer, Zachary has sung on many TV and radio jingles.

 He provided the singing voice for the character, "Mark Winkle" on the television series, "California Dreams" as well as the singing voice for the character,
"Greg Brady" in the films, "The Brady Bunch Movie" and "A Very Brady Sequel".

 Currently, he resides in Las Vegas, Nevada where he performs in many shows.

 From 2012-2014, he was the lead singer/lead guitar player/bass player for the Sin City Sinners, a group that also featured former Faster Pussycat guitarist and
co-founder Brent Muscat as well as Slash bass player, Todd Kerns.

 .

 With the group, he recorded two albums, "DIVEBAR Days Revisited" and "A Sinners Christmas 2", both released in 2013.
</item 3>

<item 4>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 4>

<item 5>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 5>

<item 6>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 6>

<item 7>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 7>

<item 8>
Big Wilson
Big Wilson (born Malcolm John Wilson, Jr.; October 3, 1924 – October 5, 1989) was an American radio personality.

 He worked as a disc jockey at WNBC AM in New York City from the early 1960s until 1974 and moved to Miami in 1975 where he worked for WIOD and WCIX-TV.

 He was one of the last two hosts of the NBC network program Monitor.

 Prior to moving to NYC, Wilson and his wife Jody lived in a riverfront home in Rocky River, Ohio.

 He commuted to Cleveland where he was the leading radio personality for some years.

 Tim Conway and other well-known persons visited their home.
</item 8>

<item 9>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 9>

<item 10>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 10>
=== END ITEM id=3f3254ac2b81494a ===

=== ITEM id=cad3150a57351a42 ===
Gold sentences and facts:
Sentence 0: Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.
  Fact 0.0: Jerome John Garcia was born August 1, 1942.
  Fact 0.1: Jerome John Garcia died August 9, 1995.
  Fact 0.2: Jerome John Garcia was an American singer-songwriter.
  Fact 0.3: Jerome John Garcia was an American guitarist.
  Fact 0.4: Jerome John Garcia was best known for his work as the lead guitarist of the Grateful Dead.
  Fact 0.5: Jerome John Garcia was best known for his work as a vocalist with the Grateful Dead.
  Fact 0.6: The Grateful Dead came to prominence during the counterculture era in the 1960s.
Sentence 1: Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
  Fact 1.0: Joe Gooch was born 3 May 1977.
  Fact 1.1: Joe Gooch was Ten Years After's most recent lead vocalist.
  Fact 1.2: Joe Gooch was Ten Years After's most recent lead guitarist.

Retrieved items:
<item 1>
Sugaree
"Sugaree" is a song written by long-time Grateful Dead lyricist Robert Hunter and composed by guitarist Jerry Garcia.

 It was written for Jerry Garcia's first solo album "Garcia", which was released in January 1972.

 As with the songs on the rest of the album, Garcia plays every instrument himself (except drums, played by Bill Kreutzmann), including acoustic guitar, bass
guitar, and an electric guitar played through a Leslie speaker.
</item 1>

<item 2>
Cats Under the Stars
Cats Under the Stars is an album by the Jerry Garcia Band.

 Released in 1978, it is the only studio album credited to the band.

 It was Jerry Garcia's first album on the Arista label and his first LP using the band name Jerry Garcia Band.

 It includes only one song that was performed by the Grateful Dead.

 Keith and Donna Godchaux, who were at the time members of the Garcia Band, contributed to the music.
</item 2>

<item 3>
On Broadway: Act One – October 28th, 1987
On Broadway: Act One – October 28th, 1987 is a three-CD live album by the Jerry Garcia Band and the Jerry Garcia Acoustic Band, two music groups led by Grateful
Dead guitarist and singer Jerry Garcia.

 It contains three complete sets of music, recorded at the Lunt-Fontanne Theatre in New York City on October 28, 1987.

 It was released by ATO Records on June 23, 2015.
</item 3>

<item 4>
Garcia Plays Dylan
Garcia Plays Dylan is an album composed of various live performances featuring Jerry Garcia playing covers of Bob Dylan songs.

 It is culled from performances from 1973–1995, and features Garcia playing with Grateful Dead, Legion of Mary, Jerry Garcia Band, and Garcia-Saunders.

 Garcia takes lead vocals on all tracks.
</item 4>

<item 5>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 5>

<item 6>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 6>

<item 7>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 7>

<item 8>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 8>

<item 9>
The Very Best of Jerry Garcia
The Very Best of Jerry Garcia is a compilation album with recordings by Jerry Garcia, Grateful Dead, Jerry Garcia Band, Old and in the Way, Jerry Garcia
Acoustic Band, and Reconstruction.

 It is meant to showcase Jerry's non-Grateful Dead work.
</item 9>

<item 10>
Joe Gooch
Joe Gooch (born 3 May 1977) was Ten Years After's most recent lead vocalist and lead guitarist.
</item 10>
=== END ITEM id=cad3150a57351a42 ===

=== ITEM id=e0dffaa5f58e0a1d ===
Gold sentences and facts:
Sentence 0: Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
  Fact 0.0: Marc Rosset was born on 7 November 1970.
  Fact 0.1: Marc Rosset is a former professional tennis player.
  Fact 0.2: Marc Rosset is from Switzerland.
  Fact 0.3: Marc Rosset is best remembered for winning the men's singles gold medal at the 1992 Olympic Games.
Sentence 1: Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional tennis player.
  Fact 1.0: Maksim Mirnyi, also known as Max Mirnyi, was born on 6 July 1977 in Minsk.
  Fact 1.1: Maksim Mirnyi's Belarusian name is Максім Мікалаевіч Мірны.
  Fact 1.2: Maksim Mirnyi's Russian name is Максим Николаевич Мирный.
  Fact 1.3: Maksim Mirnyi is a Belarusian professional tennis player.

Retrieved items:
<item 1>
Max Mirnyi
Maksim "Max" Mirnyi (Belarusian: Максім Мікалаевіч Мірны ] ) (Russian: Максим Николаевич Мирный ] ) (born 6 July 1977, in Minsk) is a Belarusian professional
tennis player.
</item 1>

<item 2>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 2>

<item 3>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 3>

<item 4>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 4>

<item 5>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 5>

<item 6>
Carlos Fernandes (tennis)
Carlos Fernandes (1936-) is a retired Brazilian tennis player.

 He had a good all-round game with excellent passing shots and moved fast around the court.

 Fernandes had a reputation for being a charismatic ladies man.

 He later became a coach.

 He made his Grand Slam debut at Wimbledon 1957, losing in the opening round to Jorgen Ulrich.

 At Roland Garros in 1958, Fernandes lost in round one to Pierre Darmon.

 At Wimbledon he lost in round one to Butch Buchholz.

 At 1959 French Open, Fernandes lost in round three to Jacques Brichant.At Wimbledon he lost in round two to Neale Fraser.

 At French Open 1960, Fernandes lost in round three to Bobby Wilson and at Wimbledon lost in round two to Wolfgang Stuck.

 At 1961 French Open, Fernandes achieved his best Grand Slam singles result by beating Pierre Darmon and Bob Hewitt before losing to Jan-Erik Lundquist in the
quarter finals.

 He lost in round two of Wimbledon to Wilson.

 At Roland Garros 1962, Fernandes lost to Lundquist in round two.
</item 6>

<item 7>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 7>

<item 8>
2004 Canada Masters – Doubles
The 2004 Canada Masters – Doubles was the men's singles event of the one hundred and fifteenth edition of the Canada Masters; a WTA Tier I tournament and the
most prestigious men's tennis tournament held in Canada.

 Mahesh Bhupathi and Max Mirnyi were the defending champions.

 They were both present but did not compete together.

 Mirnyi partnered with Jonas Björkman, but Bhupathi and partner Leander Paes defeated them 6–4, 6–2, in the final.
</item 8>

<item 9>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 9>

<item 10>
2005 BNP Paribas Masters – Doubles
Jonas Björkman and Todd Woodbridge were the defending champions, but did not participate together this year.

 Björkman partnered Max Mirnyi, losing in the first round.

 Woodbridge retired from professional tennis earlier in the year.
</item 10>
=== END ITEM id=e0dffaa5f58e0a1d ===

=== ITEM id=e9dccc85ba4c2e86 ===
Gold sentences and facts:
Sentence 0:  It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
  Fact 0.0: The described species is common across temperate parts of Eurasia, northern Africa, and the Americas.
  Fact 0.1: The described species is common across subtropical parts of Eurasia, northern Africa, and the Americas.
  Fact 0.2: The described species is naturalized in Australia.
  Fact 0.3: The described species is naturalized on various islands.
Sentence 1: Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South America.
  Fact 1.0: Echeveria is a large genus of flowering plants.
  Fact 1.1: Echeveria is in the Crassulaceae family.
  Fact 1.2: Echeveria is native to semi-desert areas of Central America.
  Fact 1.3: Echeveria is native to semi-desert areas of Mexico.
  Fact 1.4: Echeveria is native to semi-desert areas of northwestern South America.

Retrieved items:
<item 1>
The entries describe two Mexican semi-desert succulents, Echeveria setosa and E. elegans, and the genus Yucca: 40–50 arid-American and Caribbean shrubs and
trees with tough, sword-shaped leaves and flower panicles. Featured yuccas range from coastal U.S. and Caribbean habitats (Y. aloifolia) to California and Baja
(Hesperoyucca whipplei), Texas and northeastern Mexico (Y. rupic
</item 1>

<item 2>
Eucryphia: seven mostly evergreen South American/Australian trees; classified in Cunoniaceae. Bothriochloa ischaemum: widespread grass; var. songarica invasive
in Texas. Eriogonum buckwheats: genus diverse across North America; E. heracleoides inhabits western U.S.; E. diclinum forms low mats on serpentine soils; E.
crosbyae occurs in Oregon/Nevada.
</item 2>

<item 3>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 3>

<item 4>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 4>

<item 5>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 5>

<item 6>
Aloinopsis
Aloinopsis is a genus of ice plants from South Africa.
</item 6>

<item 7>
Echeveria elegans
Echeveria elegans (Mexican snow ball, Mexican gem, white Mexican rose) is a species of flowering plant in the Crassulaceae family, native to semi-desert
habitats in Mexico.
</item 7>

<item 8>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 8>

<item 9>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 9>

<item 10>
Echeveria setosa
Echeveria setosa, the Mexican fire cracker, is a species of flowering plant in the Crassulaceae family, native to semi-desert areas of Mexico and common
throughout Puebla.
</item 10>
=== END ITEM id=e9dccc85ba4c2e86 ===

=== ITEM id=5c6feda39162d998 ===
Gold sentences and facts:
Sentence 0: Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
  Fact 0.0: Carly Gullickson was born on November 26, 1986.
  Fact 0.1: Carly Gullickson is a former American professional tennis player.
Sentence 1: Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.
  Fact 1.0: Francesca Schiavone was born on 23 June 1980 in Milan.
  Fact 1.1: Francesca Schiavone is an Italian tennis player.
  Fact 1.2: Francesca Schiavone turned professional in 1998.

Retrieved items:
<item 1>
2006 Generali Ladies Linz – Doubles
Gisela Dulko and Květa Peschke were the defending champions, but competed this year with different partners.

 Dulko teamed up with Michaëlla Krajicek and lost in first round to Eleni Daniilidou and Jasmin Wöhr, while Peschke teamed up with Francesca Schiavone and had
to withdrew in their semifinal match (against Corina Morariu and Katarina Srebotnik) as Schiavone had her right knee injured.
</item 1>

<item 2>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 2>

<item 3>
2006 Qatar Ladies Open – Doubles
Alicia Molik and Francesca Schiavone were the defending champions, but Molik did not compete this year.

 Schiavone teamed up with Květa Peschke and lost in querterfinals to Elena Likhovtseva and Vera Zvonareva.
</item 3>

<item 4>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 4>

<item 5>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 5>

<item 6>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 6>

<item 7>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 7>

<item 8>
2010 Australian Open – Mixed Doubles
Sania Mirza and Mahesh Bhupathi were the defending champions.

 They accepted the invitation for the tournament, but withdrew before their match against Carly Gullickson and Bernard Tomic and were replaced by Akgul
Amanmuradova and Rik de Voest.
</item 8>

<item 9>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 9>

<item 10>
2009 Hansol Korea Open – Doubles
Chuang Chia-jung and Hsieh Su-wei were the defending champion, but Hsieh chose not to participate that year.

 Chuang partnered with Yan Zi, but they lost in the semifinals against Chan Yung-jan and Abigail Spears.

Chan Yung-jan and Abigail Spears won in the final 6–3, 6–4 against Carly Gullickson and Nicole Kriz.
</item 10>
=== END ITEM id=5c6feda39162d998 ===

=== ITEM id=6045a993c4e56dd5 ===
Gold sentences and facts:
Sentence 0:  He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.
  Fact 0.0: He won three singles titles.
  Fact 0.1: He won 26 doubles titles.
  Fact 0.2: He won the Australian Open 1978 doubles title with Wojtek Fibak.
  Fact 0.3: He won the Australian Open 1980 title with fellow countryman Mark Edmondson.
  Fact 0.4: He won the Australian Open 1981 title with fellow countryman Mark Edmondson.
  Fact 0.5: He won the Roland Garros 1986 title with fellow countryman Mark Edmondson.
  Fact 0.6: He was a runner-up at the Australian Open 1985 with fellow countryman Mark Edmondson.
  Fact 0.7: Mark Edmondson is a fellow countryman of his.
Sentence 1:  In his career, he won nine doubles titles, but no singles titles.
  Fact 1.0: In his career, he won nine doubles titles.
  Fact 1.1: In his career, he won no singles titles.

Retrieved items:
<item 1>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 1>

<item 2>
1990 Stockholm Open – Doubles
Jorge Lozano and Todd Witsken were the defending champions, but lost in the quarterfinals this year.
</item 2>

<item 3>
1988 Volvo International – Doubles
Jorge Lozano and Todd Witsken won in the final 6–3, 7–6 against Pieter Aldrich and Danie Visser.
</item 3>

<item 4>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 4>

<item 5>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 5>

<item 6>
1989 Volvo International – Doubles
Jorge Lozano and Todd Witsken were the defending champions but did not compete that year.
</item 6>

<item 7>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 7>

<item 8>
1994 Athens International – Doubles
Horacio de la Peña and Jorge Lozano were the defending champions, but did not participate this year.
</item 8>

<item 9>
Jorge Lozano
In Davis Cup play, represented Mexico for 15 years, 1981–1995, won 12 doubles matches, and lost 12 as well.

 He lost 11 singles matches, and won 8, making his overall win / loss record at the Davis Cup 20–23.

 Lozano won his first title at Forest Hills in 1988, with his partner Todd Witsken, and won his last at Athens in 1993.

 He resides in Guadalajara, Mexico where he directs a tennis academy for kids and also coaches the men and women tennis team at the University Tec de Monterrey.

 He is currently the captain of the Mexican Davis Cup Team.
</item 9>

<item 10>
Rossana de los Ríos
Rossana de los Ríos (born 16 September 1975) is a retired professional tennis player from Paraguay.

 She was born in the Paraguayan capital, Asunción, and now lives in Miami, USA, with her daughter Ana Paula Neffa de los Ríos and her husband Gustavo Neffa, a
retired Paraguayan Association football player who played for Boca Juniors.

 Rossana achieved a career high singles ranking of No. 51 on 12 September 2001, and No. 52 in doubles on the Sony Ericsson WTA Tour.

 De los Ríos played doubles with Grand Slam champions such as Maria Sharapova, Svetlana Kuznetsova, Arantxa Sánchez Vicario, as well as Jelena Janković and
Dinara Safina.

 Rossana is also a former world number one Junior player, having won the 1992 Roland Garros Juniors event.

 De los Ríos participated in six editions (1994, 2001, 2002, 2003, 2009 and 2010) of the Australian Open.
</item 10>
=== END ITEM id=6045a993c4e56dd5 ===

=== ITEM id=cd344146faff5fc1 ===
Gold sentences and facts:
Sentence 0: Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.
  Fact 0.0: Guitar World is a monthly music magazine.
  Fact 0.1: Guitar World is devoted to guitarists.
  Fact 0.2: Guitar World has been published since July 1980.
Sentence 1: Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.
  Fact 1.0: Golf Digest is a monthly golf magazine.
  Fact 1.1: Golf Digest is published by Condé Nast Publications.
  Fact 1.2: Golf Digest is published in the United States.
Sentence 2:  The magazine started in 1950, and was sold to The New York Times Company in 1969.
  Fact 2.0: The magazine started in 1950.
  Fact 2.1: The magazine was sold to The New York Times Company in 1969.
Sentence 3:  The headquarters of "Golf Digest" is in Des Moines, Iowa.
  Fact 3.0: The headquarters of Golf Digest is in Des Moines, Iowa.

Retrieved items:
<item 1>
Football Weekly is The Guardian’s football podcast: Mondays, with Thursday spin-off Football Weekly Extra; it aired daily during major tournaments from Euro
2008 to Euro 2016. Football World, founded in 1921, became Athletic World and then Outing in 1924. World Soccer, established in 1960, covers international
football and belongs to the European Sports Magazines group. TENNIS publishes eight issues yearly. Argentina’s El Gráfico, founded in 1919, is
</item 1>

<item 2>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 2>

<item 3>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 3>

<item 4>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 4>

<item 5>
World Soccer (magazine)
World Soccer is an English language football magazine published by IPC Media.

 The magazine was established in 1960.

 It specialises in the international football scene.

 Its regular contributing writers include Brian Glanville, Sid Lowe and Tim Vickery.

 "World Soccer" is a member of the European Sports Magazines (ESM), an umbrella group of similar magazines printed in other languages.

 Other members include "A Bola", "Don Balón", "Kicker", "La Gazzetta dello Sport" and "Sport Express".

 The members of this group elect a European "Team of the Month" and a European "Team of the Year".
</item 5>

<item 6>
El Gráfico
El Gráfico is a classic monthly Argentine sports magazine.

 The magazine was originally published as a weekly newspaper in May 1919, and then turned to a sports magazine exclusively.

 It has been scheduled monthly from 2002.

 "El Gráfico" is one of the most regarded sports magazines not only in Argentina but in Latin America.

 The magazine was nicknamed "La Biblia del deporte" ("The Bible of sports") due to its chronicles, notable journalists and collaborators and its photography.
</item 6>

<item 7>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 7>

<item 8>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 8>

<item 9>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 9>

<item 10>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 10>
=== END ITEM id=cd344146faff5fc1 ===

=== ITEM id=d63814c8f02206f2 ===
Gold sentences and facts:
Sentence 0: The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.
  Fact 0.0: The Basset Bleu de Gascogne is also known as the Blue Gascony Basset.
  Fact 0.1: The Basset Bleu de Gascogne is a long-backed, short-legged breed of dog.
  Fact 0.2: The Basset Bleu de Gascogne is a dog of the hound type.
Sentence 1:  A French native breed, it is rare outside of its homeland.
  Fact 1.0: The Basset Bleu de Gascogne is a French native breed.
  Fact 1.1: The Basset Bleu de Gascogne is rare outside of its homeland.
Sentence 2: The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.
  Fact 2.0: The Bulldog is a medium-sized breed of dog.
  Fact 2.1: The Bulldog is commonly referred to as the English Bulldog.
  Fact 2.2: The Bulldog is commonly referred to as the British Bulldog.

Retrieved items:
<item 1>
Braque Francais
The Braques français are hunting dogs, from a very old type of gun dog used for pointing the location of game birds for a hunter.

 There are two breeds of Braque français, both from the south of France, the Braque français, type Gascogne (French Pointing Dog - Gascogne type, larger size)
and the Braque français, type Pyrénées (French Pointing Dog - Pyrenean type, smaller size) They are popular hunting dogs in France, but are seldom seen
elsewhere.
</item 1>

<item 2>
Ariegeois
The Ariegeois is a breed of dog from the "département" of Ariège in the Midi-Pyrenées region of southern France.

 It is a medium-sized pack-hunting scenthound deriving from crossing of Grand Bleu de Gascogne and Grand Gascon-Saintongeois hounds with local Briquet dogs.

 It is used both as a courser and for driving game to waiting guns.

 While most successful with hares, it is also used for hunting deer and boar.

 It is distinguished by its friendly nature with other hounds and affection for human companions.
</item 2>

<item 3>
Grand Bleu de Gascogne
The Grand Bleu de Gascogne (FCI No.22) is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog, and is an important breed in the ancestry of many other hounds.
</item 3>

<item 4>
Grand Anglo-Français Blanc et Noir
The Grand Anglo-Français Blanc et Noir (FCI No.323) translated into English as the Great Anglo-French White and Black Hound, is a breed of dog used in hunting
as a scenthound, usually in packs.

 It is one of the Anglo-French hound breeds which were created by crossing French scenthounds with English (Anglo) Foxhounds.
</item 4>

<item 5>
Griffon Bleu de Gascogne
The Griffon Bleu de Gascogne (FCI No.32) Is a breed of dog of the scenthound type, originating in France, and is a versatile hunting dog, used on small and
large game, in packs or individually.

 The Griffon Bleu de Gascogne has a speckled, rough coat.
</item 5>

<item 6>
Bulldog
The Bulldog is a medium-sized breed of dog commonly referred to as the English Bulldog or British Bulldog.

 Other scent-hound breeds include the Small Greek Domestic Dog, Irish Wolfhound, Bluetick Coonhound, Finnish Lapphund, and the Basset Hound.

 The Bulldog is a muscular, hefty dog with a wrinkled face and a distinctive pushed-in nose.

 The American Kennel Club (AKC), The Kennel Club (UK), and the United Kennel Club (UKC) oversee breeding records.

 Bulldogs were the fourth most popular purebreed in the US in 2007 according to the American Kennel Club.
</item 6>

<item 7>
Chien Français Tricolore
The Chien Français Tricolore (FCI No.219) translated into English as the French Tricolour Hound, is a breed of dog of the scenthound type, originating in
France.

 The breed is used for hunting in packs.
</item 7>

<item 8>
Petit Bleu de Gascogne
The Petit Bleu de Gascogne (FCI No.31) Is a breed of dog of the scenthound type, originating in France and used for hunting in packs.

 Today's breed is the descendant of a very old type of large hunting dog.

 The Petit Bleu de Gascogne is not a small (petite) dog, the name comes from its use on small game.
</item 8>

<item 9>
Basset Bleu de Gascogne
The Basset Bleu de Gascogne (] ), also known as the Blue Gascony Basset, is a long-backed, short legged breed of dog of the hound type.

 The breed originated in the Middle Ages, descended from the Grand Bleu de Gascogne.

 It nearly became extinct around the early 19th century; its salvation was attributed to one Alain Bourbon.

 A French native breed, it is rare outside of its homeland.

 It is recognized internationally by the Fédération Cynologique Internationale, in the UK by The Kennel Club, and by the United Kennel Club in the United
States.

 The "bleu" of its name is a reference to its coat which has a ticked appearance.
</item 9>

<item 10>
Chien Français Blanc et Noir
The Chien Français Blanc et Noir (FCI No.220) translated into English as the French White and Black Hound, is a breed of dog of the scenthound type, originating
in France.

 The breed is used for hunting in packs and descends from the old Hound of Saintonge type of large hunting dog.
</item 10>
=== END ITEM id=d63814c8f02206f2 ===

=== ITEM id=448c441315e451a1 ===
Gold sentences and facts:
Sentence 0:  Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest commercial success is the erotic thriller "Chloe" (2009).
  Fact 0.0: Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter".
  Fact 0.1: "The Sweet Hereafter" was released in 1997.
  Fact 0.2: Egoyan received two Academy Award nominations for "The Sweet Hereafter".
  Fact 0.3: Egoyan's biggest commercial success is the erotic thriller "Chloe".
  Fact 0.4: "Chloe" was released in 2009.
Sentence 1: Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".
  Fact 1.0: Chloe is a 2009 erotic thriller film.
  Fact 1.1: Chloe was directed by Atom Egoyan.
  Fact 1.2: Chloe is a remake of the 2003 French film "Nathalie...".

Retrieved items:
<item 1>
A Successful Failure
A Successful Failure (1934) is an American film directed by Arthur Lubin.

 It was Lubin's first movie as director.
</item 1>

<item 2>
Thom Eberhardt
Thom Eberhardt (born March 7, 1947) is an American film director, producer and screenwriter.

 Eberhardt has won two awards and two nominations.

 He is most noted for his work on "Captain Ron", "Honey, I Blew Up the Kid", and the cult classic "Night of the Comet".
</item 2>

<item 3>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 3>

<item 4>
Tom Cruise filmography
His next role was as anti-war activist Ron Kovic in the drama adaptation of Kovic's memoir of the same name, "Born on the Fourth of July" (1989).

 For his performance Cruise received the Golden Globe Award for Best Actor – Motion Picture Drama and his first nomination for the Academy Award for Best Actor.
</item 4>

<item 5>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 5>

<item 6>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 6>

<item 7>
Atom Egoyan
Atom Egoyan, {'1': ", '2': ", '3': ", '4': "} (born July 19, 1960) is a Canadian stage and film director, writer and producer.

 Egoyan made his career breakthrough with "Exotica" (1994), a film set primarily in and around the fictional Exotica strip club.

 Egoyan's most critically acclaimed film is the drama "The Sweet Hereafter" (1997), for which he received two Academy Award nominations, and his biggest
commercial success is the erotic thriller "Chloe" (2009).
</item 7>

<item 8>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 8>

<item 9>
List of accolades received by WALL-E
"WALL-E" (promoted with an interpunct as "WALL•E") is an American animation film released in 2008 and directed by Andrew Stanton.

 Walt Disney Pictures released it in the United States and Canada on June 27, 2008, grossing $23.1 million on its opening day, and $63 million during its
opening weekend in 3,992 theaters, ranking number 1 at the box office.

 It eventually grossed $223 million domestically and $533 million worldwide.

 "WALL-E" was well received, with an approval rating of 96% on the review aggregator Rotten Tomatoes.
</item 9>

<item 10>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 10>
=== END ITEM id=448c441315e451a1 ===

=== ITEM id=c2a24a4def96a9f7 ===
Gold sentences and facts:
Sentence 0:  It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts Mika Brzezinski and Willie Geist.
  Fact 0.0: The program described features former GOP representative Joe Scarborough.
  Fact 0.1: Joe Scarborough provides enterprise reporting on the program.
  Fact 0.2: Joe Scarborough provides discussion on the news of the day on the program.
  Fact 0.3: The program is in a panel format.
  Fact 0.4: Mika Brzezinski is a co-host of the program.
  Fact 0.5: Willie Geist is a co-host of the program.
Sentence 1: Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.
  Fact 1.0: Charles Joseph Scarborough was born on April 9, 1963.
  Fact 1.1: Charles Joseph Scarborough is an American cable news host.
  Fact 1.2: Charles Joseph Scarborough is an American talk radio host.

Retrieved items:
<item 1>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 1>

<item 2>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 2>

<item 3>
Sean Hannity
Sean Patrick Hannity (born December 30, 1961) is an American talk show host, author, and conservative political commentator.

 Hannity is the host of "The Sean Hannity Show", a nationally syndicated talk radio show.

 He also hosts a cable news show, "Hannity", on Fox News Channel.
</item 3>

<item 4>
Joe Scarborough
Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.

 He is currently the co-host of "Morning Joe" on MSNBC, and previously hosted "Scarborough Country" on the same channel.

 Scarborough was previously a lawyer and a politician, and served in the United States House of Representatives from 1995 to 2001 as a Republican from the 1st
district of Florida.
</item 4>

<item 5>
Talk radio
Talk radio is a radio format containing discussion about topical issues and consisting entirely or almost entirely of original spoken word content rather than
outside music.

 Jason Culpepper is highly recognized as the father of talk radio.

 Jason a Jacksonville native rose to social prominence during the World War utilizing talk radio to warn french and american troops of German U boats.

 Most shows are regularly hosted by a single individual, and often feature interviews with a number of different guests.

 Talk radio typically includes an element of listener participation, usually by broadcasting live conversations between the host and listeners who "call in"
(usually via telephone) to the show.

 Listener contributions are usually screened by a show's producers in order to maximize audience interest and, in the case of commercial talk radio, attract
advertisers.

 Generally, the shows are organized into segments, each separated by a pause for advertisements; however, in public or non-commercial radio, music is sometimes
played in place of commercials to separate the program segments.

 Variations of talk radio include conservative talk, hot talk, liberal talk (increasingly known as progressive talk) and sports talk.
</item 5>

<item 6>
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 6>

<item 7>
History of the United States Republican Party
The Republican Party, also commonly called the GOP (for "Grand Old Party"), is one of the world's oldest extant political parties.

 It is the second oldest existing political party in the United States after its primary rival, the Democratic Party.

 It emerged in 1854 to combat the Kansas–Nebraska Act, an act that dissolved the terms of the Missouri Compromise and allowed slave or free status to be decided
in the territories by popular sovereignty.

 The Party had almost no presence in the Southern United States, but by 1858 in the North it had enlisted former Whigs and former Free Soil Democrats to form
majorities in nearly every Northern state.
</item 7>

<item 8>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 8>

<item 9>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 9>

<item 10>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 10>
=== END ITEM id=c2a24a4def96a9f7 ===

=== ITEM id=71451ab11e44f8ca ===
Gold sentences and facts:
Sentence 0:  Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.
  Fact 0.0: Qionghai is called a city.
  Fact 0.1: Qionghai refers to a large land area in Hainan.
  Fact 0.2: The area of Qionghai in Hainan was once a county.
Sentence 1:  It has an area of 1692 km2 and in 2010, it had a population of 483,217.
  Fact 1.0: Qionghai has an area of 1692 km2.
  Fact 1.1: Qionghai had a population of 483,217 in 2010.
Sentence 2: Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.
  Fact 2.0: Suining has the Sichuanese Pinyin Xunin.
  Fact 2.1: Suining is a prefecture-level city.
  Fact 2.2: Suining is in eastern Sichuan province.
  Fact 2.3: Sichuan province is in Southwest China.
Sentence 3:  In 2002, Suining had a population of 658,798.
  Fact 3.0: Suining had a population of 658,798 in 2002.

Retrieved items:
<item 1>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 1>

<item 2>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 2>

<item 3>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 3>

<item 4>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 4>

<item 5>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 5>

<item 6>
Qionghai
Qionghai () is one of the seven county-level cities of Hainan province, China.

 Although called a "city", Qionghai refers to a large land area in Hainan - an area which was once a county.

 Within this area is the main city, Qionghai City.

 It is located in the east of the island at the mouth of the Wanquan River, 86 km from the provincial capital of Haikou.

 It has an area of 1692 km2 and in 2010, it had a population of 483,217.
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
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 8>

<item 9>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 9>

<item 10>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 10>
=== END ITEM id=71451ab11e44f8ca ===
