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

This file has 10 items.

=== ITEM id=d8fc81bc939d36e5 ===
Gold sentences and facts:
Sentence 0: Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
  Fact 0.0: Scott Treleaven is a Canadian artist.
  Fact 0.1: Scott Treleaven's work employs a variety of media.
  Fact 0.2: The media employed in Scott Treleaven's work include collage, film, video, drawing, photography and installation.
Sentence 1: Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.
  Fact 1.0: Edmund Mortimer's dates are 21 August 1874 to 21 May 1944.
  Fact 1.1: Edmund Mortimer was an American actor.
  Fact 1.2: Edmund Mortimer was an American film director.

Retrieved items:
<item 1>
Rob Bottin
Robin R. Bottin (born April 1, 1959) is an American special make-up effects creator.

 Known for his collaborations with directors John Carpenter, Paul Verhoeven and David Fincher, Bottin worked with Carpenter on both "The Fog" and "The Thing",
with Verhoeven on "RoboCop", "Total Recall" and "Basic Instinct", and with Fincher on "Se7en" and "Fight Club".

 His other film credits include "Legend", "Innerspace" and "Fear and Loathing in Las Vegas".
</item 1>

<item 2>
Tommy Atkins (director)
Tommy Atkins was an American director of the silent and early sound film eras.

 Born on July 18, 1887 in Springfield, Massachusetts, he would make his entrance into the film industry as the assistant director to Ralph Ince on the 1920
silent film, "Out of the Snows".

 It would be another eight years before he would make another film, again as assistant director, this time for FBO Pictures, on another silent film, "Crooks
Can't Win".

 He'd work as the assistant director on another sixteen films between 1928 and 1934, the most notable of which would be 1933's "Morning Glory", directed by
Lowell Sherman and starring Katharine Hepburn and Douglas Fairbanks Jr..

 In 1934 he would be given the chance to helm his first picture, "The Silver Streak", which was one of the top money-makers for RKO Pictures that year.

 He would only direct two more films, the second of which, "Hi, Gaucho!

", he would also write the story for.
</item 2>

<item 3>
Richard Estes
Richard Estes (born May 14, 1932 in Kewanee, Illinois) is an American artist, best known for his photorealist paintings.

 The paintings generally consist of reflective, clean, and inanimate city and geometric landscapes.

 He is regarded as one of the founders of the international photo-realist movement of the late 1960s, with such painters as John Baeder, Ralph Goings, Chuck
Close, Audrey Flack, and Duane Hanson.

 Author Graham Thompson writes "One demonstration of the way photography became assimilated into the art world is the success of photorealist painting in the
late 1960s and early 1970s.

 It is also called super-realism or hyper-realism and painters like Richard Estes, Denis Peterson, Audrey Flack, and Chuck Close often worked from photographic
stills to create paintings that appeared to be photographs."
</item 3>

<item 4>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 4>

<item 5>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 5>

<item 6>
Edmund Mortimer (actor)
Edmund Mortimer (21 August 1874 – 21 May 1944) was an American actor and film director.

 He appeared in 251 films between 1913 and 1945.

 He also directed 23 films between 1918 and 1928.

 He was born in New York, New York and died in Los Angeles, California.
</item 6>

<item 7>
Scott Treleaven
Scott Treleaven is a Canadian artist whose work employs a variety of media including collage, film, video, drawing, photography and installation.
</item 7>

<item 8>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 8>

<item 9>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 9>

<item 10>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 10>
=== END ITEM id=d8fc81bc939d36e5 ===

=== ITEM id=b0b34aca9e3b4aed ===
Gold sentences and facts:
Sentence 0:  The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.
  Fact 0.0: The album includes a collaboration with Popcaan.
  Fact 0.1: The album includes a collaboration with Zhu.
  Fact 0.2: The album includes a collaboration with Leikeli47.
  Fact 0.3: The album includes a collaboration with Dreezy.
  Fact 0.4: The album includes a collaboration with Flume.
  Fact 0.5: The album includes a collaboration with Pell.
Sentence 1: Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of 2014, signed to Mind of a Genius Records and Columbia Records.
  Fact 1.0: Steven Zhu was born in 1989.
  Fact 1.1: Steven Zhu is known professionally as Zhu.
  Fact 1.2: Zhu is a Chinese American electronic musician.
  Fact 1.3: Zhu is a Chinese American singer.
  Fact 1.4: Zhu has been active since the beginning of 2014.
  Fact 1.5: Zhu is signed to Mind of a Genius Records.
  Fact 1.6: Zhu is signed to Columbia Records.

Retrieved items:
<item 1>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 1>

<item 2>
Vektroid
Ramona Andra Xavier (born August 19, 1992, known by her primary alias Vektroid) is an American electronic musician from Portland, Oregon.

 She has released music under her primary Vektroid alias, as well as others, such as dstnt, Laserdisc Visions, New Dreams Ltd., Macintosh Plus, Virtual
Information Desk, and PrismCorp Virtual Enterprises.

 Xavier played a prominent role in the popularization of the vaporwave subgenre with the release of her only studio album under her Macintosh Plus alias,
"Floral Shoppe", in 2011.

 The album helped popularize the vaporwave genre throughout the Internet.

 Since then, she has continued to release music through Bandcamp and other online platforms, and has recently collaborated with rapper Siddiq on two albums,
"Midnight Run" and "Palacio Del Rio" (both from 2016).
</item 2>

<item 3>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 3>

<item 4>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 4>

<item 5>
Illmind
Ramon Ibanga, Jr., professionally known as Illmind (often stylized as !

llmind) is a Filipino American hip hop producer and educator from Bloomfield, New Jersey.

 He has co-written and produced hip-hop, pop, soul, and electronic music for labels such as Cash Money, Def Jam, and Aftermath, and has worked with artists such
as Andy Mineo, Kanye West, Drake, J. Cole, Dr. Dre, Ludacris, 50 Cent, Ariana Grande, and others.

 His accomplishments in 2016 include production on the new Hamilton Mixtape and Disney's Moana (2016 film), both created by Lin-Manuel Miranda.

 !

llmind has released several studio albums and had a number of songs and albums reach the music charts.

 His most recent album, "Human", is a collaboration with Joell Ortiz, and was released on July 17, 2015.

 Starting in 2008 !

llmind has released the free mixtape series "Blaps, Rhymes & Life," and he operates a producer showcase in New York city called Beats, Love, Alcohol & Party
(B.L.A.P.).
</item 5>

<item 6>
I Remember (AlunaGeorge album)
I Remember is the second studio album by English electronic music duo AlunaGeorge.

 It was released on 16 September 2016 by Island Records.

 The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.

 The album's lead single, "I'm in Control", was released in February 2016, featuring vocals from rapper Popcaan, and peaked at number 39 on the UK Singles
Chart.
</item 6>

<item 7>
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 7>

<item 8>
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
</item 8>

<item 9>
Candy Chen
Candy Chen (陳斯亞 born January 3, 1993) is a dancer, actress, host, singer, rapper and a model.

 She is featured on the Taiwanese variety shows: Blackie's Teenage Club and Blackie Lollipop.

 Candy was a member of the Taiwanese girl group Hey Girl from 2010 to 2011, and a member of from 2013 to 2016.

 She is 1/8th Dutch.

 She changed her name to Chen Yi-Ling in July 2012 to follow her mother's maiden name.

 Her English name becomes Nina Chen.

 In November 2013, she changed her name to Candy Chen (English Name) and Chen Si Ya (Chinese Name).
</item 9>

<item 10>
Kid Confucius
Kid Confucius were an eight-piece Australian band from Sydney, Australia.

 They formed in 2001 and have since played well over 400 live shows around the country, including major festivals as well as their own headline shows at venues
such as The Annandale Hotel and The Metro Theatre.

 In 2005 the band released a self-titled album and two singles "Words" and "Skintight."

 The album, a mish-mash of soul, hip-hop, pop and funk, received great critical acclaim, most notably from Rolling Stone who hailed the album as one of the
standout local releases of the year.

 "Words" also enjoyed some solid months of radio and TV play.

 The band released its second album Stripes in 2007 with three singles "Closer", "Last Straw" and "Moment".

 Stripes was the band's attempt at making a Detroit-era Motown soul album and it was quick to earn rave reviews from press around the country as well as a
publishing deal with Mushroom.

 "Moment" has enjoyed good radio play on triple j. Kid Confucius is set to release its third album in October 2008.

 The first single from this album, "Good Luck", is out now.
</item 10>
=== END ITEM id=b0b34aca9e3b4aed ===

=== ITEM id=aa6f511794a55af0 ===
Gold sentences and facts:
Sentence 0:  U2 did not play the song live until 2 July 2009, when they performed it on the second concert of their U2 360° Tour in Barcelona.
  Fact 0.0: U2 did not play the song live until 2 July 2009.
  Fact 0.1: U2 performed the song on 2 July 2009 on the second concert of their U2 360° Tour in Barcelona.
Sentence 1: The U2 360° Tour was a worldwide concert tour by rock band U2.
  Fact 1.0: The U2 360° Tour was a worldwide concert tour.
  Fact 1.1: The U2 360° Tour was by rock band U2.
Sentence 2:  The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.
  Fact 2.0: The U2 360° Tour concerts featured the band playing 'in the round'.
  Fact 2.1: The U2 360° Tour concerts used a circular stage.
  Fact 2.2: The circular stage allowed the audience to surround the band on all sides.

Retrieved items:
<item 1>
Psycho Circus World Tour
Psycho Circus World Tour was a Kiss concert tour in 1998–2000.

 It was the first concert tour in history to have 3-D visual effects.

 The Smashing Pumpkins opened at the Dodger Stadium show only, in costume as The Beatles for the Halloween night performance.

 The Dodger Stadium show was streamed live on the internet as well as a radio broadcast.

 Two songs, "Psycho Circus" and "Shout It Out Loud", were screened live on Fox television as part of the "Kiss Live: The Ultimate Halloween Party" special.

 The vast majority of songs in the setlist were played on the previous Alive/Worldwide concert dates, leading to some frustration from fans expecting the return
of classic songs not played on the previous tour.

 Peter Criss was quoted in Metal Edge magazine at the time as wanting to add "Parasite" to the setlist.

 The tour was initially hyped as having circus-style acts as pre-show entertainment.

 This ultimately happened only at the first concert at Dodger Stadium.
</item 1>

<item 2>
Rock &amp; Roll Over Tour
Rock and Roll Over Tour is a concert tour by the American hard rock group Kiss.

 It began November 24, 1976 (shortly after the release of the Rock and Roll Over album) and ended April 4, 1977.

 It was on this tour where the band performed for the first time at Madison Square Garden on 2/18/77 - a venue all four members of the band had long dreamed of
playing (February 18, 1977).

 The band also performed in Japan for the first time on this tour.

 They played Budokan Hall four nights in a row breaking the previous concert attendance record set by The Beatles.

 On December 12, 1976 Ace Frehley suffered a severe electrical shock on stage which delayed the concert.

 The delay lasted about fifteen minutes and then resumed; Ace was able to finish the show despite having lost feeling in one of his hands.

 Being inspired by the event, Ace wrote the song "Shock Me" which appeared on Kiss' next album release Love Gun (1977).

 It is the first Kiss song in which Ace sings lead vocals.
</item 2>

<item 3>
U2 360° Tour
The U2 360° Tour was a worldwide concert tour by rock band U2.

 Launched in support of the group's 2009 album "No Line on the Horizon", the tour visited stadiums from 2009 through 2011.

 The concerts featured the band playing "in the round" on a circular stage, allowing the audience to surround them on all sides.

 To accommodate the stage configuration, a large four-legged structure nicknamed "The Claw" was built above the stage, with the sound system and a cylindrical,
expanding video screen on top of it.

 At 164 ft tall, it was the largest stage ever constructed.

 U2 claimed that the tour would be "the first time a band has toured in stadiums with such a unique and original structure."
</item 3>

<item 4>
Psycho Circus World Tour
They used a translator on stage to let the crowd understand exactly what Stanley was saying.

 At the end of the performance, the band ignited all of the pyrotechnics at once; as a result, they were banned from performing in Bremen.

 Ticket sales for this tour were notably slower than the previous Reunion Tour, with many of the smaller market shows underselling and a second North American
leg for the summer of 1999 cancelled all together, the band ultimatly decided on embarking on a Farewell tour in the new millennium.
</item 4>

<item 5>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 5>

<item 6>
World on Fire World Tour
The World on Fire World Tour is the third concert tour by lead guitarist Slash as a solo artist, which started in July 2014 and was scheduled to resume in late
2015, in support of Slash's third solo album "World on Fire".

 The tour features the same backing band that performed with Slash during his two last tours, billed as "Slash featuring Myles Kennedy and The Conspirators",
featuring Myles Kennedy handling lead vocal, bassist Todd Kerns, drummer Brent Fitz and rhythm guitarist Frank Sidoris.
</item 6>

<item 7>
Escape Tour
The Escape Tour was a concert tour by the American rock band Journey.

 It was one of Journey's most successful concert tours, in support of their first number one album, "Escape".

 The tour included six consecutive sold out dates at the Pine Knob Theatre in Detroit, and four straight sold out shows at the Los Angeles Forum, and Chicago's
Rosemont Horizon.

 Journey also made an appearance on July 2, 1982 at the Rose Bowl in Pasadena, California with Blue Öyster Cult, Triumph and Aldo Nova.

 The 8 ⁄ -month tour took Journey through Japan and North America.

 Point Blank, Billy Squier, the Greg Kihn Band and Loverboy were the opening acts during the tour.
</item 7>

<item 8>
The Year of the Voyager
The Year Of The Voyager is a double DVD/CD released by Nevermore.

 It was released in Europe on October 20, 2008 and in North America on November 25, 2008 via Century Media.

 The set covers the "This Godless Endeavor" touring cycle, starting with live footage from the U.S. Gigantour 2005, the Metal Mania festival 2006 in Poland, the
Wacken Open Air festival in 2006 in Germany and the main DVD show recorded at the Zeche in Bochum, Germany.

 Bonus material includes two songs from Century Media USA 10th Anniversary Party 2001, all promo videos and an interview with singer Warrel Dane which was
recorded at the Roax Film Studios in Berlin in the spring of 2008.
</item 8>

<item 9>
Psycho Circus World Tour
This ultimately happened only at the first concert at Dodger Stadium.

 Peter Criss later said that it didn't work out because the circus performers wanted equal billing and that some had even wanted to use KISS' backstage dressing
room.

 "2,000 Man" was played to bring in the new Millennium at the 1999/2000 New Year's Eve show at Vancouver, advertised at the time as being recorded for Alive IV.

 The Vancouver show was also notable as being the first time the original members had played non-original band era material live in concert – "I Love It Loud",
"Lick It Up" and "Heaven's on Fire" were added to the setlist and subsequently played on the Farewell Tour.

 "Forever" was listed on concert setlists at the Vancouver show but was not played.

 It was thought at the time it may have been intended as a Paul Stanley solo version prior to "Black Diamond".

 One notable show on the tour was the March 12 Bremen, Germany, show.

 After the opening song, Paul Stanley announced that the local fire marshall had banned Kiss from using any pyrotechnics during the show.
</item 9>

<item 10>
Because We Can (concert tour)
Because We Can was a concert tour by American rock band, Bon Jovi.

 The tour was in support of the band's twelfth studio album "What About Now".

 The tour was named after the lead single from "What About Now".

 All five major continents were visited during the 2013 tour.

 The tour saw the band travel to Wales for the first time since their 2001 One Wild Night Tour and was also the first time since their 1995 These Days Tour that
the band returned to Africa for two dates in South Africa.

 Furthermore, this tour was the first time in the band's 30-year history that they visited Bulgaria and Poland.

 The tour ranked 1st on Pollstar's annual "Top 100 Mid Year Worldwide Tours".

 It earned $142.1 million from 60 shows.

 At the end of 2013, the tour placed 1st on Pollstar's "Top 100 Worldwide Tours", grossing $259.5 million from 102 shows.
</item 10>
=== END ITEM id=aa6f511794a55af0 ===

=== ITEM id=be1cc56f58f0e6de ===
Gold sentences and facts:
Sentence 0: Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and "".
  Fact 0.0: Steve Rash is an American film director.
  Fact 0.1: Steve Rash is an American producer.
  Fact 0.2: Steve Rash is best known for directing The Buddy Holly Story.
  Fact 0.3: Steve Rash is best known for directing Can't Buy Me Love.
  Fact 0.4: Steve Rash is best known for directing Queens Logic.
  Fact 0.5: Steve Rash is best known for directing two further films whose titles are blank in the source text.
Sentence 1: Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.
  Fact 1.0: Can't Buy Me Love is a 1987 American film.
  Fact 1.1: Can't Buy Me Love is a teen romantic comedy feature film.
  Fact 1.2: Can't Buy Me Love stars Patrick Dempsey.
  Fact 1.3: Can't Buy Me Love stars Amanda Peterson.
  Fact 1.4: The story of Can't Buy Me Love is about a nerd at a high school in Tucson, Arizona.
  Fact 1.5: In Can't Buy Me Love, the nerd gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

Retrieved items:
<item 1>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 1>

<item 2>
Steve Rash
Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and
"".
</item 2>

<item 3>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 3>

<item 4>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 4>

<item 5>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 5>

<item 6>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 6>

<item 7>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 7>

<item 8>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 8>

<item 9>
Bring It On: In It to Win It
Bring It On: In It to Win It is a 2007 teen film directed by Steve Rash and starring Ashley Benson, Cassie Scerbo and Michael Copon.
</item 9>

<item 10>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 10>
=== END ITEM id=be1cc56f58f0e6de ===

=== ITEM id=d5920a25855431e7 ===
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
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 1>

<item 2>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 2>

<item 3>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 3>

<item 4>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 4>

<item 5>
Admirals Nunatak
Admirals Nunatak ( ) is a nunatak rising to 925 m on the upper Uranus Glacier, central Alexander Island, Antarctica.

 The name originates from dog teams named "The Admirals" that served at various British stations in Antarctica, 1952–94, and honors the loyal service of all
Falkland Islands Dependencies Survey/BAS sled dogs.

 The nunatak appears to have some relation to Huns Nunatak which lies about 3.7 mi northeast of Admirals Nunatak.
</item 5>

<item 6>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 6>

<item 7>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 7>

<item 8>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
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
=== END ITEM id=d5920a25855431e7 ===

=== ITEM id=2cc1e4881b9bb560 ===
Gold sentences and facts:
Sentence 0:  He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.
  Fact 0.0: The person finished second in voting for the Heisman Trophy in 1956.
  Fact 0.1: Paul Hornung finished first in that Heisman Trophy voting in 1956.
Sentence 1: Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.
  Fact 1.0: Paul Vernon Hornung was born on December 23, 1935.
  Fact 1.1: Paul Vernon Hornung is nicknamed "The Golden Boy".
  Fact 1.2: Paul Vernon Hornung is a former professional American football player.
  Fact 1.3: Paul Vernon Hornung is a Hall of Fame running back.
  Fact 1.4: Paul Vernon Hornung played for the Green Bay Packers of the National Football League (NFL).
  Fact 1.5: Paul Vernon Hornung played for the Green Bay Packers from 1957 to 1966.
  Fact 1.6: Paul Vernon Hornung won four NFL titles.
  Fact 1.7: Paul Vernon Hornung won the first Super Bowl.

Retrieved items:
<item 1>
Terry Baker
Terry Wayne Baker (born May 5, 1941) is a former American football and basketball player.

 He played college football and college basketball at the Oregon State University.

 He played as a quarterback for the Oregon State Beavers football team from 1960 to 1962, winning the Heisman Trophy as senior.

 In the spring of his senior year, he played in the Final Four of the 1963 NCAA Men's Division I Basketball Tournament with the Oregon State Beavers men's
basketball team.

 To date, he is the only athlete to win a Heisman Trophy and play in the Final Four.

 Baker was the first overall pick in the 1963 NFL draft and played with the Los Angeles Rams of the National Football League (NFL) from 1963 to 1965.

 He then played for one season in the Canadian Football League (CFL) with the Edmonton Eskimos, in 1966.

 Baker was inducted into the College Football Hall of Fame in 1982.
</item 1>

<item 2>
1976 NCAA Division I football season
The 1976 NCAA Division I football season ended with a championship for the Panthers of the University of Pittsburgh.

 Led by head coach Johnny Majors (voted the AFCA Coach of the Year), the Pitt Panthers brought a college football championship to the home of the defending pro
football champions, the Steelers.

 Pitt also had the Heisman Trophy winner, Tony Dorsett; the Panthers had been ranked ninth in the preseason AP poll.
</item 2>

<item 3>
2012 Texas A&amp;M vs. Alabama football game
The 2012 Texas A&M vs. Alabama football game was a college football game between the Texas A&M Aggies and Alabama Crimson Tide at Bryant–Denny Stadium in
Tuscaloosa, Alabama.

 This was the game where Texas A&M freshman quarterback and eventual Heisman Trophy winner Johnny Manziel completed 24 of 31 passes, had 253 passing yards, and
two passing touchdowns with 92 rushing yards to help No. 15 Texas A&M upset No. 1 Alabama 29–24, which led him to being the first freshman to win the Heisman
Trophy.
</item 3>

<item 4>
Frank Eliscu
Frank Eliscu (1912–1996) was an American sculptor and art teacher who designed and created the Heisman Memorial Football Trophy in 1935 when he was only 20
years old.

 The first Heisman Trophy, a strong young bull of a football player cast in bronze, was presented to a college football player in 1935, and is considered one of
the greatest honors a college athlete can receive.

 Over the years the Heisman Committee has paid tribute to Eliscu and his creation several times, and in 1985, the Heisman Committee invited him to speak at the
50th Anniversary Heisman ceremonies.
</item 4>

<item 5>
Paul Hornung
Paul Vernon Hornung (born December 23, 1935), nicknamed "The Golden Boy", is a former professional American football player, a Hall of Fame running back for the
Green Bay Packers of the National Football League (NFL) from 1957 to 1966, winning four NFL titles and the first Super Bowl.

 He is the first in pro football history to win the Heisman Trophy, be selected as the first overall selection in the NFL Draft, win the NFL most valuable
player award, and be inducted into both the professional and college football halls of fame.
</item 5>

<item 6>
Johnny Bright
Johnny D. Bright (June 11, 1930 – December 14, 1983) was a professional Canadian football player in the Canadian Football League.

 He played college football at Drake University.

 He is a member of the Canadian Football Hall of Fame, the National Football Foundation's College Football Hall of Fame, the Missouri Valley Conference Hall of
Fame, the Edmonton Eskimos Wall of Honour, the Alberta Sports Hall of Fame, and the "Des Moines Register's" Iowa Sports Hall of Fame.
</item 6>

<item 7>
Heisman Trophy
The Heisman Memorial Trophy (usually known colloquially as the Heisman Trophy or The Heisman), is awarded annually to the most outstanding player in college
football in the United States whose performance best exhibits the pursuit of excellence with integrity.

 Winners epitomize great ability combined with diligence, perseverance, and hard work.

 It is presented by the Heisman Trophy Trust in early December before the postseason bowl games.
</item 7>

<item 8>
Johnny Majors
John Terrill Majors (born May 21, 1935) is a former American football player and coach.

 A standout halfback at the University of Tennessee, he was an All-American in 1956 and a two-time winner of the Southeastern Conference Most Valuable Player
award, in 1955 and 1956.

 He finished second to Paul Hornung in voting for the Heisman Trophy in 1956.

 Majors served as the head football coach at Iowa State University (1968–1972), the University of Pittsburgh (1973–1976, 1993–1996), and Tennessee (1977–1992),
compiling a career college football record of 185–137–10.

 His 1976 Pittsburgh squad won a national championship after capping a 12–0 season with a victory in the Sugar Bowl.

 Majors was inducted into the College Football Hall of Fame as a player in 1987.
</item 8>

<item 9>
Charlie Ward
Charlie Ward Jr. (born October 12, 1970) is a retired American professional NBA basketball player, college football Heisman Trophy winner, Davey O'Brien Award
winner and a Major League Baseball draftee.

 Despite his NCAA football success, Ward was one of the very few players who won a Heisman trophy but was not drafted in the NFL draft.

 He won the College Football National Championship with the Florida State University Seminoles.

 Ward played several years with the New York Knicks and started in the NBA Finals.

 He was inducted in the College Football Hall of Fame in 2006.

 An avid tennis player, Ward also displayed his skills at the Arthur Ashe Tennis Tournament in 1994.
</item 9>

<item 10>
1997 Tennessee Volunteers football team
However, the Vols' defense could not stop Nebraska's rushing attack, giving up more than 400 yards on the ground in a 42–17 loss.

 As a senior, Manning won numerous awards.

 He was a consensus first-team All-American and won the Maxwell Award, the Davey O'Brien Award, the Johnny Unitas Award, and the Best College Football Player
ESPY Award, among others.

 However, he did not win the Heisman Trophy, finishing runner-up to Charles Woodson, a CB from Michigan, and the only defensive player ever to win the Heisman
Trophy.
</item 10>
=== END ITEM id=2cc1e4881b9bb560 ===

=== ITEM id=eceb341b19863c4c ===
Gold sentences and facts:
Sentence 0: The University of Delaware (colloquially "UD") is the largest university in Delaware.
  Fact 0.0: The University of Delaware is colloquially called 'UD'.
  Fact 0.1: The University of Delaware is the largest university in Delaware.
Sentence 1: The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.
  Fact 1.0: The University of Debrecen's Hungarian name is 'Debreceni Egyetem'.
  Fact 1.1: The University of Debrecen is a university.
  Fact 1.2: The University of Debrecen is located in Debrecen, Hungary.

Retrieved items:
<item 1>
Palace of Music (Miskolc)
The Palace of Music ("Zenepalota") is a building in Bartók square, Miskolc, Hungary.

 It is the building of the Béla Bartók Secondary School and the Béla Bartók Music Institute (a faculty of the University of Miskolc.)

 The Palace was designed by Gyula Waelder in Neo-baroque style and was built between 1926 and 1927.

 The construction was financed from USA loans, just like that of the Hotel Palace in Lillafüred and the Market Hall on Búza tér.
</item 1>

<item 2>
University of Delaware Botanic Gardens
The University of Delaware Botanic Gardens are botanical gardens and an arboretum located on the campus of the University of Delaware, in Newark, Delaware,
United States.

 The gardens are open to the public without charge.
</item 2>

<item 3>
University of Debrecen
The University of Debrecen (Hungarian: "Debreceni Egyetem" ) is a university located in Debrecen, Hungary.

 It is the oldest continuously operating institution of higher education in Hungary (since 1538).

 The university has a well established programme in the English language for international students, particularly in the Medical field, which first established
education in English in 1986.

 There are nearly 4000 international students studying at the university.
</item 3>

<item 4>
Delaware–William &amp; Mary football rivalry
The Delaware–William & Mary football rivalry between the Delaware Fightin' Blue Hens and the William & Mary Tribe is a match-up between two public universities,
the University of Delaware and the College of William and Mary, that are also members of the Colonial Athletic Association.

 Both schools have academic reputations that have labeled them as Public Ivies.

 Both schools are also Colonial Colleges having been founded before the United States became independent in 1776; the College of William and Mary was founded in
1693 and the University of Delaware's predecessor school was founded in 1743.
</item 4>

<item 5>
Commonwealth System of Higher Education
The Commonwealth System of Higher Education is a statutory designation by the Commonwealth of Pennsylvania that confers "state-related" status on four
universities located within the state.

 The designation establishes the schools as an "instrumentality of the commonwealth" and provides each university with annual, non-preferred financial
appropriations in exchange offering tuition discounts to students that are residents of Pennsylvania and a minority state-representation on each school's board
of trustees.

 Legally, however, the universities remain separate and private entities, operating under their own charters, governed by independent boards of trustees, and
with its assets under their own ownership and control thereby retaining much of the freedom and individuality of private institutions, both administratively and
academically.

 It is the only public-private hybrid system of higher education in the United States that is so construed, although Cornell University, the University of
Delaware, and Rutgers University represent alternative types of public-private university hybrids.
</item 5>

<item 6>
University of Delaware
The University of Delaware (colloquially "UD") is the largest university in Delaware.

 The main campus is in Newark, with satellite campuses in Dover, Wilmington, Lewes, and Georgetown.

 It is considered a large institution with approximately 18,500 undergraduate and 4,500 graduate students. UD is a privately governed university which receives
public funding for being a land-grant, sea-grant, space-grant and urban-grant state-supported research institution.
</item 6>

<item 7>
Lake Fehér (Szeged)
Lake Fehér is a lake of Hungary.

 It is named Fehér (White), because it is white.

 It is a part of Kiskunság National Park, and it is situated just north of the town of Szeged.

 It covers an area of 14 square kilometres.

 Lake Fehér is Hungary's largest saltwater lake.

 It is carefully protected because it is home to 280 species of bird and because of its ancient flora.

 The area is an important stop in the migration of European birds, and serves as a resting and feeding place for them.

 During the migration season, crowds of cranes, wild geese, and teal varieties are visible on Lake Fehér.
</item 7>

<item 8>
College Park, Delaware
College Park is an unincorporated community in New Castle County, Delaware, United States.

 College Park is located along Delaware Route 896 southwest of Newark.
</item 8>

<item 9>
Pike Creek, Delaware
Pike Creek is a census-designated place (CDP) in New Castle County, Delaware and is part of the Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metropolitan
statistical area.

 In 2007, CNN's Money Magazine ranked Pike Creek on its list of the 100 Best Places to Live in the United States.

 Pike Creek was the only community in Delaware to appear on the list.

 The population was 7,898 at the 2010 census.

 Local attractions include Middle Run Natural Area, White Clay Creek State Park, Carousel Farm Park & Equestrian Center, Golf and Restaurants.

 Pike Creek is approximately 5 miles from downtown Newark, Delaware which contains the University of Delaware.

 Pike Creek is connected to the University of Delaware campus and the Newark, Delaware downtown area by several bike trails.

 Pike Creek is approximately 10 miles west of the state's largest city, Wilmington, Delaware which is a national financial center.

 Chester County, Pennsylvania is located approximately 2 miles west of Pike Creek.
</item 9>

<item 10>
Rutgers University–Camden
Rutgers University–Camden is one of three regional campuses of Rutgers University, New Jersey's public research university.

 It is located in Camden, New Jersey, United States.

 Founded in the 1920s, Rutgers–Camden began as an amalgam of the South Jersey Law School and the College of South Jersey.

 It is the southernmost of the three regional campuses of Rutgers—the others being located in New Brunswick and Newark.

 The city of Camden is located on the Delaware River, east of Philadelphia.
</item 10>
=== END ITEM id=eceb341b19863c4c ===

=== ITEM id=8f0da3de44907689 ===
Gold sentences and facts:
Sentence 0:  He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink, T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
  Fact 0.0: This producer has produced songs for Justin Bieber.
  Fact 0.1: This producer has produced songs for Monica.
  Fact 0.2: This producer has produced songs for Britney Spears.
  Fact 0.3: This producer has produced songs for Usher.
  Fact 0.4: This producer has produced songs for Keri Hilson.
  Fact 0.5: This producer has produced songs for T.I.
  Fact 0.6: This producer has produced songs for Nelly Furtado.
  Fact 0.7: This producer has produced songs for Kevin Cossom.
  Fact 0.8: This producer has produced songs for Ciara.
  Fact 0.9: This producer has produced songs for Mariah Carey.
  Fact 0.10: This producer has produced songs for Timbaland.
  Fact 0.11: This producer has produced songs for Madonna.
  Fact 0.12: This producer has produced songs for J. Cole.
  Fact 0.13: This producer has produced songs for Kanye West.
  Fact 0.14: This producer has produced songs for Whitney Houston.
  Fact 0.15: This producer has produced songs for Missy Elliott.
  Fact 0.16: This producer has produced songs for M.I.A.
  Fact 0.17: This producer has produced songs for Justin Timberlake.
  Fact 0.18: This producer has produced songs for JoJo.
  Fact 0.19: This producer has produced songs for Joe Jonas.
  Fact 0.20: This producer has produced songs for Simple Plan.
  Fact 0.21: This producer has produced songs for The Clutch.
  Fact 0.22: This producer has produced songs for Pink.
  Fact 0.23: This producer has produced songs for T-Pain.
  Fact 0.24: This producer has produced songs for Diddy.
  Fact 0.25: This producer has produced songs for Meek Mill.
  Fact 0.26: This producer has produced songs for Bjork.
  Fact 0.27: This producer has produced songs for Duran Duran.
Sentence 1:  Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.
  Fact 1.0: Cole was raised in Fayetteville, North Carolina.
  Fact 1.1: Cole initially gained recognition as a rapper.
  Fact 1.2: Cole initially gained recognition as a rapper after the release of his debut mixtape.
  Fact 1.3: Cole's debut mixtape is titled The Come Up.
  Fact 1.4: Cole's debut mixtape The Come Up was released in early 2007.

Retrieved items:
<item 1>
Zach Crowell
Zach Crowell is a Grammy Award-winning songwriter and record producer, who was born and raised in Nashville, Tennessee.

 Crowell is involved in a joint publishing deal with Warner/Chappell Music, Combustion Music and Taperoom Music.
</item 1>

<item 2>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 2>

<item 3>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 3>

<item 4>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 4>

<item 5>
Lil Ru
Sylvester Samuels better known by his stage name Lil Ru, is an American rapper from Ridgeway, South Carolina currently signed to Def Jam Recordings.

 His debut album, 21 & Up was released on August 25, 2009.

 The Ridgeway native was 16 when he made his professional foray into the music business.

 Inspired by New Orleans’ innovative Cash Money Crew, Ru began making a name for himself on his local music scene, doing live shows and pressing up his own CDs.

 His hard-hitting lyrics and entrepreneurial spirit caught the attention of fellow South Carolinian Angie Stone.

 Shortly after the neo-soul songstress helped him secure a deal with Elektra Records, Ru found himself unsigned again, among the artists lost in the shuffle
after the label merged with Atlantic Records.

 Music fans first heard him on his 2001 debut single Will Destroy.

 He then released his 2002 follow up, Shawty What You Doin’.

 Both songs reached the Billboard R&B/Hip-Hop charts and helped land him at his next label home, Capitol Records.
</item 5>

<item 6>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 6>

<item 7>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
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
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
J. Cole
Jermaine Lamarr Cole (born January 28, 1985), better known by his stage name J. Cole, is an American hip hop recording artist and record producer.

 Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.

 Intent on further pursuing a solo career as a rapper, he went on to release two additional mixtapes after signing to Jay Z's Roc Nation imprint in 2009.
</item 10>
=== END ITEM id=8f0da3de44907689 ===

=== ITEM id=1039833db044dc5f ===
Gold sentences and facts:
Sentence 0: Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
  Fact 0.0: Stephen Lawrence 'Steve' Winwood was born 12 May 1948.
  Fact 0.1: Steve Winwood is an English musician.
  Fact 0.2: Steve Winwood's genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
Sentence 1: Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.
  Fact 1.0: Per 'Pelle' Almqvist was born 29 May 1978.
  Fact 1.1: Per 'Pelle' Almqvist is also known as Howlin' Pelle Almqvist.
  Fact 1.2: Per 'Pelle' Almqvist is a Swedish singer and songwriter.

Retrieved items:
<item 1>
Pelle Almqvist
Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.

 He is the lead singer of garage rock band The Hives.
</item 1>

<item 2>
Back in the High Life Again
"Back in the High Life Again" is a 1986 single written by Steve Winwood and Will Jennings and performed by Winwood.

 The single was included on his album "Back in the High Life" and included backing vocals by James Taylor as well as a prominent mandolin played by Winwood.

 "Back in the High Life Again" was Winwood's second number one on the Adult Contemporary chart.

 The single stayed at number one for three weeks and went to number 13 on the "Billboard" Hot 100.
</item 2>

<item 3>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 3>

<item 4>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 4>

<item 5>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 5>

<item 6>
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
</item 6>

<item 7>
Revolutions – The Very Best of Steve Winwood
Revolutions – The Very Best of Steve Winwood is the sixth compilation album by Steve Winwood.

 The album includes music from Winwood's solo career, as well as groups with which he has performed, including The Spencer Davis Group, Traffic, and Blind
Faith.

 The CD was released as a box set and a single disc.

 As of October 2014, the box set is out of print (but can be purchased digitally), while the single disc is still available.

 The songs "The Finer Things" and "Roll With It" are exclusive to the single disc version and cannot be found on the box set.
</item 7>

<item 8>
Steve Winwood (album)
Steve Winwood is the debut solo studio album by blue-eyed soulster Steve Winwood.

 It was released three years after the break-up of Traffic.

 Though the album sold moderately well in the USA, it was a commercial disappointment compared to Traffic's recent albums, peaking at number 22 on the
"Billboard" 200 album chart.

 In Winwood's home country the situation was reversed; while Traffic's recent albums had only been moderately successful in the UK, "Steve Winwood" reached
number 12 on The Official Charts.

 Island Records launched two singles from the album, "Hold On" and "Time Is Running Out", both of which failed to make the charts.
</item 8>

<item 9>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 9>

<item 10>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 10>
=== END ITEM id=1039833db044dc5f ===

=== ITEM id=6d250c94ca13116f ===
Gold sentences and facts:
Sentence 0:  She was also a member of a secret society composed exclusively of Cheyenne women.
  Fact 0.0: The woman in question was also a member of a secret society.
  Fact 0.1: The secret society was composed exclusively of Cheyenne women.
Sentence 1: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.
  Fact 1.0: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance.
  Fact 1.1: The other central institution of traditional Cheyenne Indian tribal governance is the Council of Forty-four.

Retrieved items:
<item 1>
Lean Bear
Lean Bear (Cheyenne name Awoninahku, 1813–1864), alternatively translated as Starving Bear, was a Cheyenne peace chief.

 He was a member of the Council of Forty-four, a tribal governance devoted to maintaining peace with encroaching United States settlers.

 Lean Bear's most notable peace deals include the Treaty of Fort Wise and a meeting with US President Abraham Lincoln.

 His work towards peace between his people and the American settlers in the Southern Plains were cut short when he was murdered by the 1st Colorado Cavalry
Regiment and violent retaliations ensued.
</item 1>

<item 2>
Ehyophsta
Ehyophsta (Cheyenne for "Yellow-Haired Woman") was a Cheyenne woman.

 She was the daughter of Stands-in-the-Timber who died in 1849, and she was the niece of Bad Faced Bull.

 She fought in the Battle of Beecher Island in 1868, and also fought the Shoshone that same year, where she counted coup against one enemy and killed another.

 She fought the Shoshone again in 1869.

 She was also a member of a secret society composed exclusively of Cheyenne women.

 She died in 1915.
</item 2>

<item 3>
Chief Dull Knife College
Chief Dull Knife College (originally Dull Knife Memorial College) is a small, open-admission, Native American tribal community college and land grant
institution.

 Located on the Northern Cheyenne Indian Reservation, in the city of Lame Deer, it has a current enrollment of about 141 students.

 On average, more than half of its graduates move on to four-year colleges.
</item 3>

<item 4>
Cheyenne military societies
Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.

 While council chiefs are responsible for overall governance of individual bands and the tribe as a whole, the headmen of military societies are in charge of
maintaining discipline within the tribe, overseeing tribal hunts and ceremonies, and providing military leadership.

 Historically, council chiefs selected which of the six military societies would assume these duties; after a period of time on-duty, the chiefs would select a
different society to take up the duties.
</item 4>

<item 5>
The material covers several Indigenous histories and conflicts: the Westervelt massacre killed 17 Dutch settlers, including ten Westervelts, and prompted some
to join George Rogers Clark’s militia; the Pottawatomie killings by John Brown’s abolitionists were part of Bleeding Kansas. Cheyenne topics include governance,
military societies, peace chief Lean Bear, warrior Ehyophsta, Northern Cheyenne Reservation and its Lame Deer headquarters, and Chief Dull Knife College.
Concho’s
</item 5>

<item 6>
Concho Indian Boarding School
Concho Indian Boarding School (also known as the Cheyenne-Arapaho Boarding School at Concho or Concho Indian School and home to the Concho Demonstration School)
was a boarding school for members of the Cheyenne and Arapaho Tribes and later opened to other Native American students.

 It existed from 1909 to 1983.

 It was located in central Oklahoma, approximately 1 mile south of Concho, Oklahoma and 4 miles north of El Reno, Oklahoma.

 The name of the town and school is the Spanish word for "shell" and was named for the Indian agent, Charles E. Shell.
</item 6>

<item 7>
Eugene Little Coyote
Eugene Little Coyote was the president of the Northern Cheyenne Indian Reservation from 2004 to 2007.

 He was elected in November 2004, defeating the incumbent president, Geri Small.

 However, after a conflict with reservation vice president Rick Wolfname that began in July 2007 escalated, the Northern Cheyenne tribal council declared that
it would oust Little Coyote as the tribe's president.

 Little Coyote was forcibly removed from office and arrested for trespassing in the tribal office building on December 28, 2007.

 It is worth noting that the move by the tribal council to remove Little Coyote from office was deemed unconstitutional and void by the Northern Cheyenne
Constitutional Court.

 The Constitutional Court was however overruled by Bureau of Indian Affairs' regional director Ed Parisian, and later the Assistant Secretary of Indian Affairs
Carl J. Artman.

 Their involvement in this matter set a historical precedence of the BIA directly meddling with tribal governments.
</item 7>

<item 8>
Northern Cheyenne Indian Reservation
The Northern Cheyenne Indian Reservation (Tsėhéstáno in Cheyenne, formerly named the Tongue River Indian Reservation) is home of the federally recognized
Northern Cheyenne Tribe.

 Located in southeastern Montana, the reservation is approximately 444,000 acres in size and home to approximately 5,000 Cheyenne people.

 The tribal and government headquarters are in Lame Deer, which is also the home of the annual Northern Cheyenne Pow wow.

 The reservation is bounded on the east by the Tongue River and on the west by the Crow Reservation.

 There are small parcels of non-contiguous off-reservation trust lands in Meade County, South Dakota, northeast of the city of Sturgis.

 Its timbered ridges that extend into northwestern South Dakota are part of Custer National Forest and it is approximately 40 mi east of the site of the 1876
Battle of the Greasy Grass (known to most Americans as the Battle of the Little Bighorn).
</item 8>

<item 9>
Council of Forty-four
The Council of Forty-four was one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the military societies such
as the Dog Soldiers.

 The influence of the Council of Forty-four waned in the face of internal conflict among the Cheyenne about Cheyenne policy toward encroaching white settlers on
the Great Plains, and was dealt a severe blow by the Sand Creek Massacre.
</item 9>

<item 10>
Cheyenne
The Cheyenne ( ) are one of the Indigenous peoples of the Great Plains and their language is of the Algonquian language family.

 The Cheyenne comprise two Native American tribes, the Só'taeo'o or Só'taétaneo'o (more commonly spelled as Suhtai or Sutaio) and the Tsétsêhéstâhese (also
spelled Tsitsistas).

 These tribes merged in the early 19th century.

 Today, the Cheyenne people are split into two federally recognized Nations: the Southern Cheyenne, who are enrolled in the Cheyenne and Arapaho Tribes in
Oklahoma, and the Northern Cheyenne, who are enrolled in the Northern Cheyenne Tribe of the Northern Cheyenne Indian Reservation in Montana.
</item 10>
=== END ITEM id=6d250c94ca13116f ===
