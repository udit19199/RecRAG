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

=== ITEM id=b616341f93b6c37d ===
Gold sentences and facts:
Sentence 0: Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
  Fact 0.0: Black is a Bangladeshi rock band.
  Fact 0.1: Black was formed in Dhaka, Bangladesh.
  Fact 0.2: The Bengali name of Black is ব্ল্যাক.
Sentence 1: New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.
  Fact 1.0: New Found Glory is an American rock band.
  Fact 1.1: New Found Glory was formerly known as A New Found Glory.
  Fact 1.2: New Found Glory is from Coral Springs, Florida.
  Fact 1.3: New Found Glory was formed in 1997.

Retrieved items:
<item 1>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 1>

<item 2>
Hit or Miss (New Found Glory song)
"Hit or Miss (Waited Too Long)" is the first single by New Found Glory (formerly A New Found Glory), from their 1999 debut studio album, "Nothing Gold Can
Stay".
</item 2>

<item 3>
Clearview 77
Clearview 77 was a punk rock quartet hailing from Brooklyn, NY.

 Born in the summer of 1996, CV77 was the true definition of the "local" band.

 The band has played countless shows within the East Coast with acts like Face to Face, New Found Glory, Finch, Saves the Day, The Amazing Crowns, US Bombs, Big
Wig, Murphy's Law, Alkaline Trio, The Queers, GOB, and many more.

 In their 8-year existence the band managed to finish at the top of the charts and competitions for CMJ, Discmakers, Garageband.com, mp3.com, Jim Beam,
Loudenergy, had a song aired on CNN and Good Morning America, and added to college radio stations across the US.

 The band also finished in the top 10 in 92.3 FM K-ROCK' best unsigned band of 2000 contest.

 Clearview 77 have also been featured on multiple compilations with acts like NOFX, New Found Glory, Auto Pilot Off, Agnostic Front, Midtown, ALL, Saves the
Day, Down By Law, Circle Jerks, and many more.
</item 3>

<item 4>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
</item 4>

<item 5>
Nothing Gold Can Stay (album)
Nothing Gold Can Stay is the debut studio album by American rock band New Found Glory, released on October 19, 1999 through independent record label Eulogy
Recordings.

 At the time, the band was then named "A New Found Glory", but later dropped the indefinite article "A" due to some fans struggling to find their records in
stores.

 The original pressings of the album contained samplings from several films including "The Outsiders" (1983), "Weird Science" (1985), and "That Thing You Do!

" (1996), as well as Robert Frost's poem "Nothing Gold Can Stay", after which the album is named.
</item 5>

<item 6>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 6>

<item 7>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 7>

<item 8>
Radiosurgery (album)
Radiosurgery is the seventh studio album by American rock band New Found Glory.

 It was first released on September 30, 2011 in Australia, before its wider release on October 4 through independent label Epitaph Records.

 It is the band's final studio album to feature founding guitarist Steve Klein.

 To follow up predecessor "Not Without a Fight" (2009), the band began writing new material during their stint on the 2010 Honda Civic Tour.

 After self-producing a set of demos and contacting long-term record producer Neal Avron, the band went on to record the album in Avron's home recording studio
over a period of three months in 2011.

 The quintet set out to write an album that paid homage to classic punk rock records that first inspired them to form a band during the 1990s.

 Listening extensively to the likes of early Green Day and Ramones, New Found Glory strove to create a sound that could "bridge the gap" between old and new
generations of the genre.
</item 8>

<item 9>
A Very New Found Glory Christmas
A Very New Found Glory Christmas is the third EP by American rock band New Found Glory.

 Self-produced by the band, it is a special edition Christmas-themed acoustic recording featuring two original songs and three covers.

 The EP was limited to 2,000 pressings and was released via cassette tape on December 1, 2012.

 1,000 white copies were sold on tour, and 1,000 red copies were sold in their online store.

 The band released "Nothing for Christmas" via Rdio and Spotify on December 20, after debuting songs off the EP during the tenth anniversary tour of their
"Sticks and Stones" album.

 The cover art is a parody of the 1990 blockbuster movie "Home Alone".
</item 9>

<item 10>
It's All About the Girls
It's All About the Girls is the debut EP by A New Found Glory (now New Found Glory) released on December 20, 1997 by Fiddler Records.

 It is the only release with their original drummer Joe Marino.

 A re-release, featuring revised cover artwork, was issued by Fiddler in June, 2000.
</item 10>
=== END ITEM id=b616341f93b6c37d ===

=== ITEM id=2b3e5002de2d4108 ===
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
</item 1>

<item 2>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 2>

<item 3>
Lydia Patterson Institute
Lydia Patterson Institute is a Methodist Christian college-preparatory school located in El Paso, Texas, United States.

 Founded in 1913, it offers programs for Spanish-speaking children, primarily from Juarez, to attend high school in the United States and attend an American
undergraduate university.

 All high school classes are taught in English, and the school is accredited by the Southern Association of Colleges and Schools.
</item 3>

<item 4>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 4>

<item 5>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 5>

<item 6>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 6>

<item 7>
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
</item 7>

<item 8>
Tunku Abdul Rahman University College
Tunku Abdul Rahman University College, abbreviated as TAR UC, is a non-profit, comprehensive, private university college in Malaysia.

 Named after the first Prime Minister of Malaysia, YTM Tunku Abdul Rahman Putra Al-Haj, the institution was founded in 1969 as Tunku Abdul Rahman College or TAR
College then by the Malaysian Chinese Association (MCA).

 In 2013, the college was upgraded to a university college.

 TAR UC is one of the prestigious private university college in Malaysia, renowned for its accounting programme.
</item 8>

<item 9>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 9>

<item 10>
Evangelical Presbyterian University College
The Evangelical Presbyterian University College was established in 2008 by the Evangelical Presbyterian Church, Ghana.
</item 10>
=== END ITEM id=2b3e5002de2d4108 ===

=== ITEM id=b415ac1041b3e4f6 ===
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
Basie's Beatle Bag
Basie's Beatle Bag is a 1966 studio album by Count Basie and his orchestra, arranged by Chico O'Farrill.

 Basie released a second album of Beatles songs, "Basie on the Beatles", in 1969.
</item 1>

<item 2>
Basie (album)
Basie (reissued as The Band of Distinction) is an album by pianist/bandleader Count Basie recorded in 1954 and originally released on the Clef label.

 the album should not be confused with Basie's 1958 album which became known as "The Atomic Mr. Basie".
</item 2>

<item 3>
The Atomic Mr. Basie
The Atomic Mr. Basie (originally called Basie, also known as E=MC and reissued in 1994 as The Complete Atomic Basie) is a 1958 album by Count Basie and his
orchestra.

 The album is one of Basie's most famous and is critically acclaimed.

 Allmusic gave it 5 stars, reviewer Bruce Eder saying "it took Basie's core audience and a lot of other people by surprise, as a bold, forward-looking statement
within the context of a big-band recording."

 It is included in the book "1001 Albums You Must Hear Before You Die", Will Fulford-Jones calling it "Basie's last great record."

 According to the 2015 site update of Acclaimed Music, it is the 6th most critically acclaimed album of 1958, the 25th most acclaimed of the 1950s, and the
837th most acclaimed of all time, based on an aggregation of hundreds of critics' lists from around the world.
</item 3>

<item 4>
Count Basie/Sarah Vaughan
Count Basie/Sarah Vaughan is a 1961 album by American jazz singer Sarah Vaughan, accompanied by the Count Basie Orchestra, with arrangements by Frank Foster,
Thad Jones and Ernie Wilkins.

 According to James Gavin's liner notes to the 1996 CD release, Basie himself does not perform on any of the tracks.
</item 4>

<item 5>
Basie on the Beatles
Basie on the Beatles is an album by pianist and bandleader Count Basie featuring performances recorded in late 1969 and released on the short-lived Happy Tiger
label.

 It was Basie's second album of Beatles' compositions following 1966's "Basie's Beatle Bag" and featured liner notes by Ringo Starr.
</item 5>

<item 6>
Count Basie at Newport
Count Basie at Newport is a live album by jazz musician Count Basie and his orchestra.

 It was originally issued as Verve MGV 8243 and included only the tracks 1-8 and 13.

 Tracks 9-12 originally included in "Count Basie & Joe Williams/Dizzy Gillespie & Mary Lou Williams at Newport" (Verve MGV 8244).
</item 6>

<item 7>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 7>

<item 8>
Count Basie
William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.

 His mother taught him to play the piano and he started performing in his teens.

 Dropping out of school, he learned to operate lights for vaudeville and to improvise accompaniment for silent films at a local movie theater in his home town
of Red Bank, New Jersey.

 By age 16, he increasingly played jazz piano at parties, resorts and other venues.

 In 1924, he went to Harlem, where his performing career expanded; he toured with groups to the major jazz cities of Chicago, St. Louis and Kansas City.

 In 1929 he joined Bennie Moten's band in Kansas City, and played with them until Moten's death in 1935.
</item 8>

<item 9>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 9>

<item 10>
Basie Meets Bond
Basie Meets Bond is a 1966 album by Count Basie and his orchestra.

 The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".

 The album was Basie's first for United Records, and was produced by Teddy Reig.
</item 10>
=== END ITEM id=b415ac1041b3e4f6 ===

=== ITEM id=908c511c6b8d5546 ===
Gold sentences and facts:
Sentence 0:  In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
  Fact 0.0: In Crash, there is no betting, unlike in Brag.
  Fact 0.1: In Crash, players aim to reach a total of 11 points.
  Fact 0.2: In Crash, the 11 points are gained over successive deals.
Sentence 1: Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.
  Fact 1.0: Three-card brag is a 16th-century British card game.
  Fact 1.1: Three-card brag is the British national representative of the vying or 'bluffing' family of gambling games.

Retrieved items:
<item 1>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 1>

<item 2>
Triomphe
Triomphe (French for triumph) is a card game dating from the late 15th century.

 It most likely originated in France or Spain (as triunfo) and later spread to the rest of Europe.

 When the game arrived in Italy, it shared a similar name with the pre-existing game and deck known as "trionfi" (tarot).

 While trionfi has a fifth suit that acts as permanent trumps, triomphe randomly selects one of the existing four suits as trumps.

 Another common feature of this game is the robbing of the stock.

 Triomphe became so popular that during the 16th century the earlier game of trionfi was gradually renamed tarocchi, tarot, or tarock.

 This game is the origin of the English word "trump" and is the ancestor of many trick-taking games like Euchre (via Écarté) and Whist (via Ruff and Honours).
</item 2>

<item 3>
Put (card game)
Put is an English tavern trick-taking card game first recorded in the 16th century and later castigated by 17th century moralists as one of ill repute.

 It belongs to a very ancient family of card games and clearly relates to a group known as Trut, "Truque", also "Tru", and the South American game Truco.

 Its more elaborate version is the Spanish game of Truc, which is still much played in many parts of Southern France and Spain.
</item 3>

<item 4>
Quadrille (card game)
Quadrille is a card game that was popular in the 18th century.

 A variant of the Spanish card game Ombre, it is played by four players in pairs, with a deck of 40 cards (the 8's, 9's and 10's being removed).

 By the end of the 19th century, the card game had fallen out of fashion.
</item 4>

<item 5>
Three card brag
Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.

 Brag is a direct descendant of the Elizabethan game of Primero and one of the several ancestors to poker, just varying in betting style and hand rankings.
</item 5>

<item 6>
Écarté
Écarté is a two-player card game originating from France, the word literally meaning "discarded".

 It is a trick-taking game, similar to whist, but with a special and eponymous discarding phase.

 It is closely related to Euchre, a card game played mainly in the United States.

 Écarté was popular in the 19th century, but is now rarely played.
</item 6>

<item 7>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 7>

<item 8>
Ombre
The historical importance of Ombre in the field of playing cards is the fact that it was the first card game in which a trump suit was established by bidding
rather than by the random process of turning the first card of the stock.

 This game developed from Triunfo, though it was from L'Hombre that the idea of bidding was adopted into other card games such as Skat, and Tarot, which owes
Hombre a good portion of its betting system as well.

 The game continued to be in vogue almost in every corner of Europe from the late 17th through the 18th centuries.
</item 8>

<item 9>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 9>

<item 10>
Egyptian Ratscrew
Egyptian Ratscrew (also known as Egyptian Ratkiller, Egyptian War, and by other names) is a card game of the matching family of games.

 The game is similar to the 19th century British card game Beggar-My-Neighbour, with the added concept of "slapping" cards when certain combinations are played,
similar to and perhaps borrowed from Slapjack.
</item 10>
=== END ITEM id=908c511c6b8d5546 ===

=== ITEM id=f04452a65d356475 ===
Gold sentences and facts:
Sentence 0:  Jan Kodeš and Margaret Court won the singles titles.
  Fact 0.0: Jan Kodeš won the singles titles.
  Fact 0.1: Margaret Court won the singles titles.
Sentence 1:  She is currently a Christian minister in Perth, Western Australia.
  Fact 1.0: A woman referred to in this sentence is currently a Christian minister.
  Fact 1.1: The Christian minister referred to in this sentence is based in Perth, Western Australia.

Retrieved items:
<item 1>
1972 French Open – Men's Singles
Jan Kodes was the two-time defending champion, but lost in the quarterfinals to Patrick Proisy.
</item 1>

<item 2>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 2>

<item 3>
Carlos Fernandes (tennis)
At Roland Garros 1962, Fernandes lost to Lundquist in round two.

 At Wimbledon he lost in round one to Neale Fraser's brother John.

 At 1963 French Open he lost in round one to Christian Duxin.

 In round one of Wimbledon he led a young Arthur Ashe by two sets to 0, but lost in five sets.

 At Wimbledon 1964 he lost in round two to Gene Scott.

 An ankle injury in 1964 took its' toll and Fernandes played less after that.

 As Open tennis arrived, Fernandes was reaching the end of his career.

 He lost in round one of 1968 French Open to Bernard Montrenaud.

 At Wimbledon he lost in round one to Mike Sangster.
</item 3>

<item 4>
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
</item 4>

<item 5>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 5>

<item 6>
2001 Milan Indoor – Singles
Marc Rosset was the defending champion at the Milan Indoor tennis tournament but lost in the second round to Greg Rusedski.
</item 6>

<item 7>
Margaret Court
Margaret Court {'1': ", '2': ", '3': ", '4': "} (née Smith; born 16 July 1942), also known as Margaret Smith Court, is a retired Australian tennis player and
former world No. 1.

 She is currently a Christian minister in Perth, Western Australia.

 In tennis, she amassed more major titles than any other player in history.
</item 7>

<item 8>
1970 French Open
The 1970 French Open was a tennis tournament that was held at the Stade Roland Garros in Paris in France from 25 May through 7 June 1970.

 It was the 69th edition of the French Open, the 40th to be open to foreign competitors, and the second Grand Slam of the year.

 Jan Kodeš and Margaret Court won the singles titles.
</item 8>

<item 9>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 9>

<item 10>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 10>
=== END ITEM id=f04452a65d356475 ===

=== ITEM id=41cdfcea31220028 ===
Gold sentences and facts:
Sentence 0: Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.
  Fact 0.0: Migene González-Wippler is a Puerto Rican.
  Fact 0.1: Migene González-Wippler is a new-age author.
  Fact 0.2: Migene González-Wippler is a leading expert on the Afro-Caribbean religion of Santería.
Sentence 1: Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among West African descendants.
  Fact 1.0: Santería is also known as Regla de Ochá.
  Fact 1.1: Santería is also known as La Regla de Ifá.
  Fact 1.2: Santería is also known as Lucumi.
  Fact 1.3: Santería is an Afro-American religion.
  Fact 1.4: Santería is a religion of Caribbean origin.
  Fact 1.5: Santería developed in the Spanish Empire.
  Fact 1.6: Santería developed among West African descendants.

Retrieved items:
<item 1>
Chiguata District
Chiguata is a district located in the Arequipa province, 30 km away from the city of Arequipa, Peru, on the slopes of the volcano Pikchu Pikchu (where at its
summit pre-Inca worshipped remains and artifacts have been found).

 The name "chiguata" comes from the Quechua words "chiri" meaning "cold" and "wata" meaning "year" (cold all year).

 Ancient Chiguata was inhabited by the local natives until the arrival of Spanish influence in charge of the emissary Diego Hernández de Córdova in 1540.

 Soon after, the construction of a small plaza was initiated, and according to the Dominican religious order and local sources, the plaza's "Templo del Espíritu
Santo" ("Sacred Spirit temple") along with its domes and lateral cover was not begun until 1739.

 The design of the temple is quite interesting, with two angels in feathered nightwear and their arms lifted vertically, separated by twelve fringes of
rectangular flowers which add to the whole decoration of the temple.
</item 1>

<item 2>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 2>

<item 3>
Santería
Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among
West African descendants.

 "Santeria" is a Spanish word that means the "worship of saints".

 Santería is influenced by and syncretized with Roman Catholicism.

 Its sacred language, a variety of Yoruba, is the Lucumí language.
</item 3>

<item 4>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 4>

<item 5>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 5>

<item 6>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
</item 6>

<item 7>
Ewe people
The Ewe people (Ewe: "Eʋeawó" , lit.

 "Ewe people"; or " Eʋedukɔ́ ", lit.

 "Ewe nation","Eʋenyigba" Eweland;) are an African ethnic group.

 They are the largest ethnic group in Togo (32%), the third largest ethnic group in Ghana (14%), and are a minority ethnic group in southern Benin, Ivory Coast
and Nigeria.

 They speak the Ewe language (Ewe: "Eʋegbe" ) which belongs to the Niger-Congo family of languages.

 They are related to other speakers of Gbe languages, such as, the Fon, Gen, Phla Phera, and the Aja people of Togo and Benin.
</item 7>

<item 8>
Ethnic religion
In religious studies, ethnic religion (or indigenous religion) is the expression of religion associated with a particular ethnic group.

 Ethnic religions are often distinguished from religions which claim to not be limited in ethnic or national scope, such as Christianity or Islam.

 Ethnic religions do not have to be excluded to independent religions.

 Some localized denominations of global religions are practiced solely by certain ethnic groups, with ethnic groups like the Assyrians following a unique
denominational structure of Christianity known as the Assyrian Church of the East.
</item 8>

<item 9>
Migene González-Wippler
Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.

 González-Wippler was born in Puerto Rico and has degrees in psychology and anthropology from the University of Puerto Rico and from Columbia University.

 In addition to her solid background in social sciences she has also worked as a science editor for the Interscience Division of John Wiley, the American
Institute of Physics, and the American Museum of Natural History, and as an English editor for the United Nations in Vienna, where she resided for many years.

 She is a cultural anthropologist and lectures frequently at universities and other educational institutions.

 She also has contributed extensively to the Institute of Puerto Rican Culture's collection, especially when it comes to Santeria and its practices, beliefs and
organization.
</item 9>

<item 10>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 10>
=== END ITEM id=41cdfcea31220028 ===

=== ITEM id=0091a12ab0546bd9 ===
Gold sentences and facts:
Sentence 0: Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.
  Fact 0.0: Ghost Rider is a song.
  Fact 0.1: Ghost Rider is a song by the protopunk band Suicide.
  Fact 0.2: Ghost Rider appears on the debut album of the protopunk band Suicide.
Sentence 1:  The song is based on the Marvel Comics character.
  Fact 1.0: The song Ghost Rider is based on the Marvel Comics character.
Sentence 2: Suicide is the debut album from the American rock band Suicide.
  Fact 2.0: The album Suicide is the debut album of the American rock band Suicide.
  Fact 2.1: The band Suicide is an American rock band.
Sentence 3:  It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.
  Fact 3.0: The album Suicide was released in 1977.
  Fact 3.1: The album Suicide was released on Red Star Records.
  Fact 3.2: The album Suicide was produced by Craig Leon.
  Fact 3.3: The album Suicide was produced by Marty Thau.

Retrieved items:
<item 1>
Talking Heads: 77
Talking Heads: 77 is the debut album by the American rock band Talking Heads, released in September 1977.

 The single "Psycho Killer" reached No. 92 on the Billboard Hot 100 Chart.

 In 2003, the album was ranked No. 290 on "Rolling Stone" magazine's The 500 Greatest Albums of All Time list.
</item 1>

<item 2>
Ghost Rider (Suicide song)
Ghost Rider is a song by the protopunk band Suicide appearing on their debut album.

 The song is based on the Marvel Comics character.
</item 2>

<item 3>
Radio Blast
Radio Blast is the fifth E.P. by Chicago-based pop punk band Screeching Weasel.

 The lineup for this single is the same as their 1993 album, Wiggle.

 2000 copies were initially pressed on Underdog Records.

 A later repress came out on burgundy colored vinyl and had a sticker on the sleeve that proclaimed it "not part of the limited edition".

 Both pressings are now out of print, but the two songs from this E.P. can be found on the B-sides and rarities collection, Kill the Musicians.

 The A-side is a vicious attack on the state of rock radio.

 Side B is the sad story of a girl who cannot find love and ends up committing suicide.

 "The Girl Next Door" was covered by Blink-182, which was first released on a 1998 reissue of their first demo album, "Buddha".
</item 3>

<item 4>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 4>

<item 5>
Suicide (1977 album)
Suicide is the debut album from the American rock band Suicide.

 It was released in 1977 on Red Star Records and produced by Craig Leon and Marty Thau.

 The album was recorded in four days at Ultima Sound Studios in New York and featured Martin Rev's minimalist electronics and harsh, repetitive rhythms paired
with Alan Vega's rock and roll-inspired vocals and depictions of urban life.
</item 5>

<item 6>
1979 (song)
"1979" is a song by American alternative rock band the Smashing Pumpkins.

 Released in 1996 as the second single from their third studio album, "Mellon Collie and the Infinite Sadness", "1979" was written by frontman Billy Corgan, and
features loops and samples that were uncharacteristic of previous Smashing Pumpkins songs.

 The song was written as a coming of age story by Corgan.

 In the year 1979, Corgan was 12 and this is what he considered his transition into adolescence.

 The song was popular with critics and fans; Allmusic's Amy Hanson called it a "somewhat surprising hit".

 The song was nominated for the Record of the Year and Best Rock Performance by a Duo or Group with Vocal at the 39th Annual Grammy Awards, and won an MTV Video
Music Award for Best Alternative Video.

 In 2012, it was voted the second-best Smashing Pumpkins song by Rolling Stone magazine readers.
</item 6>

<item 7>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 7>

<item 8>
A Boy in a Man's World
A Boy in a Man's World is the second studio album by Mucky Pup.

 The album was recorded at Nevessa Productions and released in 1989 through Torrid Records and Roadrunner Records.

 The album features a re-recorded version of "U-Stink-But-I-♥-U," which the band had previously won a Bloom County songwriting contest with.

 The band would also shoot their first music video for this song.

 The album also features "Batman", a song based on an urban legend that also contains a snippet of the title theme to the 1966 "Batman" TV.
</item 8>

<item 9>
Even If It Kills Me
Even if It Kills Me is the third studio album by American rock band Motion City Soundtrack.

 Produced by Ric Ocasek, Adam Schlesinger and Eli Janney, the album was released on September 18, 2007, in the United States by Epitaph Records.

 Motion City Soundtrack, based in Minneapolis, Minnesota, made a breakthrough with their second album, "Commit This to Memory", garnering praise and independent
buzz upon its 2005 release.

 Following the release, the band toured relentlessly over the next two years, during which time frontman Justin Pierre struggled with alcohol and substance
abuse.

 These addictions were infused into the writing process of "Even If Kills Me", which was recorded in early 2007 Stratosphere Studios in Chelsea and the
legendary Electric Lady Studios in Greenwich Village.
</item 9>

<item 10>
The Suicide File
The Suicide File was an American hardcore punk band from Boston, Massachusetts that formed in April 2001.

 The band wrote songs with a mostly political message, although many songs also dealt with social and personal problems.

 Most of the band's output was released on the Southern California-based hardcore label Indecision Records.

 The band reunited in June 2006 to embark on their first European Tour.

 Members of the bands are or have been affiliated with The Hope Conspiracy, Death By Stereo, When Tigers Fight, Adamantium, Give Up the Ghost/American
Nightmare, Clouds, Panic and many more.

 Most of these are bands with whom Alexander has drummed for short periods of time.

 Dave Weinberg was also known for his frequent collaborations and duets with Julie Ecker and James "Boom Boom" Auclair (cf. unpublished correspondence with the
artists).

 Despite their short tenure, The Suicide File continues to be revered within the hardcore punk community for their musical output and lasting impression on the
flourishing Boston hardcore scene.

 The band since 2006 has played a small number of successfully sized reunion sets.
</item 10>
=== END ITEM id=0091a12ab0546bd9 ===

=== ITEM id=58f58fec37a3cccc ===
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
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
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
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 5>

<item 6>
Oregon Territorial Legislature
Oregon’s Territorial Legislature was a bicameral legislative body created by the United States Congress in 1848 as the legislative branch of the government of
the Oregon Territory.

 The upper chamber Council and lower chamber House of Representatives first met in July 1849; they served as the region's legislative body until Oregon became a
state in February 1859, when they were replaced by the bicameral Oregon State Legislature.
</item 6>

<item 7>
Hajime Hirota
Hajime Hirota (広田 一 , Hirota Hajime , born October 10, 1968) is a Japanese politician of the Democratic Party and a former member of the House of Councillors in
the National Diet, having served two terms from 2004 until 2016.

 He previously served two terms in the Kōchi Prefectural Assembly from 1995 until 2001.
</item 7>

<item 8>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
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
Alaska Legislature
The Alaska Legislature is the state legislature of the U.S. state of Alaska.

 It is a bicameral institution, consisting of the lower Alaska House of Representatives, with 40 members, and the upper house Alaska Senate, with 20 members.

 There are 40 House Districts (1-40) and 20 Senate Districts (A-T).

 With a total of 60 lawmakers, the Alaska Legislature is the smallest bicameral state legislature in the United States and the second-smallest of all state
legislatures (only the 49-member unicameral Nebraska Legislature is smaller).

 There are no term limits for either chamber.
</item 10>
=== END ITEM id=58f58fec37a3cccc ===

=== ITEM id=18535177d3ffdca3 ===
Gold sentences and facts:
Sentence 0: Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on the German island of Greifswalder Oie in December 1937.
  Fact 0.0: Operation Lighthouse was the name given to a failed experimental launch.
  Fact 0.1: The Operation Lighthouse launch involved four Aggregate 3 liquid-fuel rockets.
  Fact 0.2: Wernher von Braun conducted the Operation Lighthouse launch.
  Fact 0.3: Walter Dornberger conducted the Operation Lighthouse launch.
  Fact 0.4: The Operation Lighthouse launch took place on the German island of Greifswalder Oie.
  Fact 0.5: The Operation Lighthouse launch took place in December 1937.
Sentence 1: Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War II.
  Fact 1.0: Major-General Dr. Walter Robert Dornberger was born on 6 September 1895.
  Fact 1.1: Walter Robert Dornberger died on 27 June 1980.
  Fact 1.2: Walter Robert Dornberger was a German Army artillery officer.
  Fact 1.3: Walter Robert Dornberger's career spanned World War I.
  Fact 1.4: Walter Robert Dornberger's career spanned World War II.

Retrieved items:
<item 1>
Magnus von Braun
Magnus "Mac" Freiherr von Braun (10 May 1919 – 21 June 2003) was a German chemical engineer, Luftwaffe aviator, and rocket scientist at Peenemünde, the
Mittelwerk, and after emigrating to the United States via Operation Paperclip, at Fort Bliss.

 He was the brother of Sigismund and Wernher von Braun.
</item 1>

<item 2>
Operation Lighthouse (1937)
Operation Lighthouse was the name given to the failed experimental launch of four Aggregate 3 liquid-fuel rockets by Wernher von Braun and Walter Dornberger on
the German island of Greifswalder Oie in December 1937.
</item 2>

<item 3>
Mount Von Braun
Mount Von Braun ( ) is a mountain (3,275 m) located 4 nautical miles (7 km) south of Mount Sabine in the Admiralty Mountains.

 Mapped by United States Geological Survey (USGS) from surveys and U.S. Navy air photos, 1960–63.

 Named by Advisory Committee on Antarctic Names (US-ACAN) for Wernher von Braun of the National Aeronautics and Space Administration, a visitor at McMurdo
Station, 1966–67.
</item 3>

<item 4>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 4>

<item 5>
Von Braun (crater)
von Braun, named after the rocket pioneer Wernher von Braun, is a lunar crater located near the northwestern limb of the Moon.

 It lies along the western edge of the Oceanus Procellarum, to the northeast of the crater Lavoisier.

 The northeastern rim of this crater is on the edge of the Sinus Roris, a bay feature in the northwestern part of the Oceanus Procellarum.

 Due to its proximity to the limb, this crater appears significantly foreshortened when viewed from the Earth.
</item 5>

<item 6>
Man Will Conquer Space Soon!
Man Will Conquer Space Soon!

 was the title of a famous series of 1950s magazine articles in "Collier's" detailing Wernher von Braun's plans for manned spaceflight.

 Edited by Cornelius Ryan, the individual articles were authored by such space notables of the time as Willy Ley, Fred Lawrence Whipple, Dr. Joseph Kaplan, Dr.
Heinz Haber, and von Braun.

 The articles were illustrated with paintings and drawings by Chesley Bonestell, Fred Freeman, and Rolf Klep, some of the finest magazine illustrators of the
time.
</item 6>

<item 7>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 7>

<item 8>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 8>

<item 9>
Von Braun Ferry Rocket
Von Braun Ferry Rocket was a concept design for a shuttle spacecraft that was developed by Wernher von Braun in a seminal series of early-1950s Collier's
magazine articles, Man Will Conquer Space Soon!

 by Wernher von Braun "et al." The Ferry Rocket concept has evolved over time.
</item 9>

<item 10>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 10>
=== END ITEM id=18535177d3ffdca3 ===

=== ITEM id=e48e9506d56b0a13 ===
Gold sentences and facts:
Sentence 0:  Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.
  Fact 0.0: Players compete in races.
  Fact 0.1: The race tracks are inspired by attractions such as the Haunted Mansion.
  Fact 0.2: The race tracks are inspired by attractions such as Pirates of the Caribbean.
  Fact 0.3: The race tracks are inspired by attractions such as Space Mountain.
  Fact 0.4: The race tracks are inspired by attractions such as Big Thunder Mountain Railroad.
  Fact 0.5: Players race on tracks to acquire missing parts for the park's fireworks machine.
  Fact 0.6: The park's fireworks machine was accidentally destroyed by Chip 'n' Dale.
  Fact 0.7: Chip 'n' Dale destroyed the fireworks machine while they were gathering acorns.
Sentence 1:  Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.
  Fact 1.0: Space Mountain opened on January 15, 1975.
  Fact 1.1: Space Mountain is the oldest operating roller coaster in the state of Florida.
  Fact 1.2: Space Mountain is the original version of an iconic attraction.
  Fact 1.3: The original version of Space Mountain has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide.
  Fact 1.4: The replication of Space Mountain excludes the Shanghai Disneyland Resort.

Retrieved items:
<item 1>
Wild West Express Coaster
Wild West Express Coaster is a steel roller coaster operating at Glenwood Caverns Adventure Park in Glenwood Springs, Colorado.

 Wild West Express Coaster opened to the public at Glenwood Caverns Adventure Park on May 25, 2012.
</item 1>

<item 2>
Treasure Island Resort &amp; Casino
An expansion that was completed in fall 2008 included 230 new hotel rooms, 30,000-square-foot event center and a bowling center complete with an arcade area.

 In 2015, Tado Steakhouse was constructed, Tradewinds Buffet was remodeled and the water park & spa construction began.

 The Lagoon and Wave Spa opened February 9, 2016.
</item 2>

<item 3>
Space Mountain (Magic Kingdom)
Space Mountain is an indoor dark outer space-themed steel roller coaster at the Magic Kingdom theme park at the Walt Disney World Resort in Lake Buena Vista,
Florida.

 Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction
that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.

 RCA helped fund Space Mountain's construction and sponsored the ride from 1975 to 1993; FedEx sponsored Space Mountain from 1994 to 2004.
</item 3>

<item 4>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 4>

<item 5>
Rock 'n' Roller Coaster Starring Aerosmith
Rock 'n' Roller Coaster Starring Aerosmith is an enclosed launched steel roller coaster at Disney's Hollywood Studios at the Walt Disney World Resort and at the
Walt Disney Studios Park in Disneyland Paris (where it is called "Rock 'n' Roller Coaster avec Aerosmith").

 As the attraction's name suggests, the coaster features Aerosmith members, Steven Tyler, Joe Perry, Tom Hamilton, Joey Kramer, and Brad Whitford.
</item 5>

<item 6>
Space Mountain (Disneyland)
Space Mountain (known as Hyperspace Mountain from 2015 to 2017) is an indoor, space-themed roller coaster in Tomorrowland at Disneyland in Anaheim, California.

 Opened on May 27, 1977, it was the second roller coaster built at Disneyland, and was the second of the five versions of Space Mountain built by The Walt
Disney Company.

 Its exterior façade is one of Disneyland's four "mountain" structures that serve as park landmarks.
</item 6>

<item 7>
Walt Disney World Quest: Magical Racing Tour
Walt Disney World Quest: Magical Racing Tour is a 2000 go-kart racing video game based on attractions at the Walt Disney World Resort.

 Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain
Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.

 The game was developed by Crystal Dynamics and published by Eidos Interactive.

 The Game Boy Color version was developed by Prolific Publishing.
</item 7>

<item 8>
The entries cover roller-coaster history, manufacturers, parks, and attractions. Ice slides preceded French wooden-track coasters; steel coasters are now
widespread. Mack Rides and Japan’s TOGO build amusement rides; Swiss Giovanola supplied Intamin before producing coasters such as Goliath and Titan. Pretzel’s
company built over 1,400 single-rail dark rides. Notable coasters include Bandit, Millennium Force, Disney’s Space Mountains, and White Cyclone. Other
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
Space Mountain
Space Mountain is the name of a space-themed indoor roller coaster attraction located at five of the Magic Kingdom-style Disney Parks (all except for Shanghai
Disneyland Park).

 Although all five versions of the attraction are different in nature, all have a similar domed exterior façade that is a landmark for the respective park.

 The first Space Mountain ride opened in 1975 at Walt Disney World Resort in Florida, and similar attractions were then eventually built at the other Disney
parks.
</item 10>
=== END ITEM id=e48e9506d56b0a13 ===

=== ITEM id=844fe9ce104d54a7 ===
Gold sentences and facts:
Sentence 0: Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.
  Fact 0.0: Catfish and the Bottlemen are a British rock band.
  Fact 0.1: Catfish and the Bottlemen were formed in Llandudno, Wales.
  Fact 0.2: Catfish and the Bottlemen were formed in 2007.
Sentence 1: The Lumineers are an American folk rock/Americana band based in Denver, Colorado.
  Fact 1.0: The Lumineers are an American folk rock/Americana band.
  Fact 1.1: The Lumineers are based in Denver, Colorado.
Sentence 2:  Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.
  Fact 2.0: Schultz and Fraites began writing together in 2005.
  Fact 2.1: Schultz and Fraites began performing together in 2005.
  Fact 2.2: Schultz and Fraites began writing together in Ramsey, New Jersey.
  Fact 2.3: Schultz and Fraites began performing together in Ramsey, New Jersey.
Sentence 3:  The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.
  Fact 3.0: The Lumineers emerged as one of the most popular folk rock/Americana artists.
  Fact 3.1: The Lumineers emerged during the revival of folk rock and Americana genres.
  Fact 3.2: The Lumineers emerged during the growing popularity of folk rock and Americana in the 2010s.

Retrieved items:
<item 1>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 1>

<item 2>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 2>

<item 3>
The Balcony (album)
The Balcony is the debut studio album by Welsh rock band Catfish and the Bottlemen.

 It was released on 15 September 2014 in the United Kingdom and 15 January 2015 in the United States.

 On 20 March 2015, "The Balcony" was awarded a Gold certification from the British Phonographic Industry.

 The album cover depicts outlines of two headless people pleasuring each other's genitals.

 The illustration is by New York artist Tim Lahan who originally posted it to his Flickr account in 2009 and was subsequently contacted by the band.
</item 3>

<item 4>
Billy Bibby &amp; The Wry Smiles
Billy Bibby & The Wry Smiles are a British indie rock band formed in the autumn of 2015 in the area around Llandudno, North Wales and Chester, UK.

 The band is fronted by Billy Bibby, former lead guitarist and founding member of Catfish and the Bottlemen.

 Bibby parted ways with Catfish and the Bottlemen in the summer of 2014, shortly after co-writing and recording their Platinum selling debut album, "The
Balcony".

 Soon thereafter, Billy Bibby began writing songs and playing acoustic shows in various UK cities in order to test public response to his new music.

 In the spring of 2015, Billy Bibby began seeking musicians to form a new band.

 Auditions were conducted with the help of Simon Jones, formerly of The Verve.

 Band members Rob Jones (lead guitar), Matt Bassy (bass), and Michael Pearce (drums) were selected to form the four-piece ensemble.
</item 4>

<item 5>
The Brightlights
The Brightlights are a four-piece British indie rock band from Grimsby in North East Lincolnshire.

 They released their debut single, "Inspired By", on 5 November 2007.

 The single was produced by Feeder frontman Grant Nicholas.
</item 5>

<item 6>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 6>

<item 7>
The Ride (Catfish and the Bottlemen album)
The Ride is the second studio album by British indie rock band Catfish and the Bottlemen.

 It was released on 27 May 2016 through Capitol Records.
</item 7>

<item 8>
Kathleen (song)
"Kathleen" is the third single by Welsh indie rock band, Catfish and the Bottlemen.

 The song was included in their EP, "Kathleen and the Other Three", and their debut studio album, "The Balcony".

 The single was digitally released on 18 March 2014 as a music video, and on 7 April 2014, the 7-inch version of the single was released.

 The single did not contain a B-side.
</item 8>

<item 9>
Beautiful Decay
Beautiful Decay is the second extended play by Welsh indie rock band, Catfish and the Bottlemen.

 Released on 10 March 2010, the EP contained one song that would later be featured in their debut studio album, "The Balcony".
</item 9>

<item 10>
The Ride Tour
The Ride Tour is the ongoing second concert tour by English indie rock band Catfish and the Bottlemen, in support of the group's second studio album "The Ride".

 The tour was announced in February 2016 and began on 4 April 2016, in Glasgow, United Kingdom at the O2 Academy Glasgow.

 The tour is planned to travel across Europe and North America with a total of 50 shows planned so far.

 On 22 February 2016, the band announced an April 2016 stint of shows in UK, including Glasgow, Halifax, Southend On Sea, Doncaster, London and Brighton.

 The band will also headline several festivals across the UK throughout the summer, including Liverpool's 'Sound City' in May 2016, Bristol's Summer Series in
July 2016 and Manchester's Castlebowl in July 2016.

 On 14 March 2016 it was announced that the band will play several dates in North America in June 2016.
</item 10>
=== END ITEM id=844fe9ce104d54a7 ===
