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

=== ITEM id=1044fb4f404249b1 ===
Gold sentences and facts:
Sentence 0:  She was born in Guangzhou, China.
  Fact 0.0: The woman referred to as "she" was born in Guangzhou, China.
Sentence 1:  These videos are an overview and recap of each year's viral videos, events, memes, and music.
  Fact 1.0: The videos being described are an overview of each year's viral videos, events, memes, and music.
  Fact 1.1: The videos being described are a recap of each year's viral videos, events, memes, and music.

Retrieved items:
<item 1>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 1>

<item 2>
CVX Live
CVX Live (Creator Viewer Experience Live) is an event designed to bring online media creators and their fans together.

 It was held for the first time in Orem, Utah, on August 7 to 8, 2015 Originally created by the team behind the "Bored Shorts TV" YouTube channel, the
convention is Utah's first YouTube event, gathering thousands of online YouTube creators and fans together in one place.

 One of the creators, Brett Robers, described it as “...a very unique opportunity to bring YouTube to life.”
</item 2>

<item 3>
Kevin Nalty
Kevin 'Nalts' Nalty (born May 12, 1969) is a YouTube comedian and partner based in Doylestown, Pennsylvania better known under his YouTube username Nalts.

 Nalts began on YouTube as one of the top-20 most-viewed comedy channels, and collaborates with many of today's top YouTube personalities.

 He has more than 1,000 videos which, as of April 18, 2016, have been viewed more than 297 million times on YouTube alone.

 He has been ranked as one of YouTube's Most Subscribed users.

 He is the author of "Beyond Viral: How to Attract Customers, Promote Your Brand, and Make Money with Online Video (Wiley & Sons, 2010).
</item 3>

<item 4>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 4>

<item 5>
Wengie
Wendy Ayche (born January 9 1986) known professionally as Wengie, is an Australian YouTube personality and vlogger.

 She was born in Guangzhou, China.

 She got the nickname "Wengie" at a dance class when two people gave her a nickname inspired by her Chinese name, Wén Jié.

 Since starting her channel on February 11 2013, her videos have received over 281 million views, and her channel has accumulated over 11 followers In the
January 2016 edition of "ElleGirl Japan", she was nominated as a channel to watch.

 In August 2016, she was ranked the 5th Fastest growing channel in the world.

 Ayche was featured in the annual YouTube Rewind in 2016 and was also singled out by Google as one of the top beauty creators in the Asia region.

 Ayche reached 5 million YouTube subscribers in January 2017, and her channel is currently in the 12th most subscribed How-To & Style Channel on YouTube.

 She is also currently a board member of the Internet Creators Guild, a non-profit focused on providing the protection, representation and guidance to online
creators.
</item 5>

<item 6>
YouTube Rewind
YouTube Rewind is a video series produced and created by YouTube and Portal A Interactive.

 These videos are an overview and recap of each year's viral videos, events, memes, and music.

 Each year, the number of YouTube celebrities featured in the video, as well as the presentation of the series, have increased.

 The latest episode of "YouTube Rewind" was released on December 8, 2016.
</item 6>

<item 7>
MatPat
Matthew Robert "Matt" Patrick, also known by screen name MatPat, is an American internet personality, actor, writer, and producer.

 He is best known as the creator and narrator of the YouTube webseries "Game Theory", where he comments on topics such as the logic, scientific accuracy, and
lore of various video games and the gaming industry.

 He is also known for creating the spinoff "Film Theory," centering around cinema and internet filmography.

 In 2015, Patrick created one of YouTube's first live gaming channels, "GTLive", and in 2016, he created the YouTube Red series "MatPat's Game Lab".

 As of June 2017, Patrick has amassed over 13 million subscribers and over 1.6 billion views total across his three channels.

 Between July 1, 2016 and July 31, 2016, he starred in the game show "The Runner".

 The show was produced by Ben Affleck and Matt Damon and was available on Go90.
</item 7>

<item 8>
Lilly Singh
Lilly Singh (born 26 September 1988) is a Canadian YouTube personality, vlogger, comedian, writer, and actress.

 She is better known by her YouTube username IISuperwomanII.

 Since beginning her channel in October 2010, her videos have received over 2 billion views, and her channel has accumulated over 12 million subscribers.

 In 2016, she was ranked 3rd on the "Forbes" list of the world's highest paid YouTube stars (behind Roman Atwood and PewDiePie), earning a reported $7.5 million
in 2016.

 Singh has featured in the annual YouTube Rewind every year since 2014.

 She ranked 1st on 2017 Forbes Top Influencers List in the entertainment category.
</item 8>

<item 9>
Wayne Goss (make-up artist)
Wayne Alan Goss (born 4 March 1978) is an English makeup artist, YouTube personality and entrepreneur.

 Best known for his "Wayne Goss" channel on YouTube, he is the creator and owner of the self-eponymous makeup brush line launched in September 2013.

 As of November 2016, Goss' main YouTube channel had over 2.7 million subscribers and had had 344 million views.
</item 9>

<item 10>
Issa Rae
Jo-Issa "Issa" Rae Diop (born January 12, 1985) is an American actress, writer, director, producer and web series creator.

 She is best known as the creator of the YouTube web series "Awkward Black Girl".

 Since the premiere of "Awkward Black Girl", Rae has developed her own YouTube platform where she features various content created by people of color.

 Rae's shows have garnered over 20 million views and over 260,000 subscribers on YouTube.
</item 10>
=== END ITEM id=1044fb4f404249b1 ===

=== ITEM id=cfd1a5d3dcdfff6a ===
Gold sentences and facts:
Sentence 0: Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
  Fact 0.0: Gunnar Eric Nelson was born on September 20, 1967.
  Fact 0.1: Gunnar Eric Nelson is an American musician.
  Fact 0.2: Gunnar Eric Nelson is an American singer.
  Fact 0.3: Gunnar Eric Nelson is an American songwriter.
  Fact 0.4: Gunnar Eric Nelson is an international multi-platinum recording artist.
Sentence 1: Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the Machine.
  Fact 1.0: Florence Leontine Mary Welch was born on 28 August 1986.
  Fact 1.1: Florence Leontine Mary Welch is an English musician.
  Fact 1.2: Florence Leontine Mary Welch is an English singer.
  Fact 1.3: Florence Leontine Mary Welch is an English songwriter.
  Fact 1.4: Florence Leontine Mary Welch is best known as the vocalist of the indie rock band Florence and the Machine.

Retrieved items:
<item 1>
Lungs (album)
Lungs is the debut studio album by English indie rock band Florence and the Machine, released on 3 July 2009 on Island Records.

 Frontwoman Florence Welch experimented with and honed the album's sound with bandmate Isabella Summers, while also collaborating with five record producers
including James Ford, Paul Epworth, Steve Mackey.

 The music on "Lungs" established Florence and the Machine as an internationally successful rock act, and also featured the group's successful mix of string
arrangements centered around Welch's vocal prowess.
</item 1>

<item 2>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 2>

<item 3>
Gunnar Nelson (musician)
Gunnar Eric Nelson (born September 20, 1967) is an American musician, singer, and songwriter and international multi-platinum recording artist.
</item 3>

<item 4>
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
</item 4>

<item 5>
Romeo Nelson
Iromeio "Romeo" Nelson (March 12, 1902 – May 17, 1974) was an American boogie woogie pianist whose recordings from 1929 are regarded as some of the finest, and
certainly the fastest, boogie woogie showpieces on record.
</item 5>

<item 6>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 6>

<item 7>
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
</item 7>

<item 8>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 8>

<item 9>
Sweet Nothing
"Sweet Nothing" is a song by Scottish recording artist and record producer Calvin Harris from his third studio album, "18 Months" (2012).

 It features vocals from Florence and the Machine singer Florence Welch.

 Harris previously worked with the band in a remix of their single, "Spectrum" (2012).

 While recording "18 Months", Harris expressed interest in working with Welch.

 He mentioned that it was not easy and he had to chase her, due to schedule conflicts.

 Welch accepted his invitation and the two recorded "Sweet Nothing".

 The track premiered on "The Chris Moyles Show" on BBC Radio 1 on 28 August 2012 and was later released as the fifth single from the album on 12 October 2012.
</item 9>

<item 10>
Florence and the Machine
Florence and the Machine (stylised as Florence + the Machine) are an English indie rock band that formed in London in 2007, consisting of lead singer Florence
Welch, keyboardist Isabella Summers, and a collaboration of other artists.

 The band's music received praise across the media, especially from the BBC, which played a large part in their rise to prominence by promoting Florence and the
Machine as part of "BBC Introducing".

 At the 2009 Brit Awards they received the Brit Awards "Critics' Choice" award.

 The band's music is renowned for its dramatic and eccentric production and also Welch's powerful vocal performances.
</item 10>
=== END ITEM id=cfd1a5d3dcdfff6a ===

=== ITEM id=c644659bb9db9a35 ===
Gold sentences and facts:
Sentence 0: Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in England to Australian parents.
  Fact 0.0: Ralph Foster Smart was born on 27 August 1908.
  Fact 0.1: Ralph Foster Smart died on 12 February 2001.
  Fact 0.2: Ralph Foster Smart was a film producer.
  Fact 0.3: Ralph Foster Smart was a television producer.
  Fact 0.4: Ralph Foster Smart was a director.
  Fact 0.5: Ralph Foster Smart was a writer.
  Fact 0.6: Ralph Foster Smart was born in England.
  Fact 0.7: Ralph Foster Smart's parents were Australian.
Sentence 1: Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.
  Fact 1.0: Randolph Severn 'Trey' Parker III was born on October 19, 1969.
  Fact 1.1: Randolph Severn 'Trey' Parker III is an American actor.
  Fact 1.2: Randolph Severn 'Trey' Parker III is an animator.
  Fact 1.3: Randolph Severn 'Trey' Parker III is a writer.
  Fact 1.4: Randolph Severn 'Trey' Parker III is a director.
  Fact 1.5: Randolph Severn 'Trey' Parker III is a producer.
  Fact 1.6: Randolph Severn 'Trey' Parker III is a singer.
  Fact 1.7: Randolph Severn 'Trey' Parker III is a songwriter.

Retrieved items:
<item 1>
Everett Greenbaum
Everett Greenbaum (December 20, 1919 – July 11, 1999) was an American television and film writer and actor who contributed to such shows as "The Andy Griffith
Show" (24 Episodes), "M*A*S*H" (35 Episodes), "Love American Style", "The Real McCoys" (32 Episodes), "Sanford and Son", and "The George Gobel Show".

 Greenbaum was a co-creator with Jim Fritzell of "Mister Peepers" an important early television show which starred Wally Cox.

 He wrote the Hollywood feature film "Good Neighbor Sam", as well as a series of films starring Don Knotts that included "The Shakiest Gun in the West", "The
Reluctant Astronaut", and "The Ghost and Mr. Chicken".
</item 1>

<item 2>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 2>

<item 3>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
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
Ralph Smart
Ralph Foster Smart {'1': ", '2': ", '3': ", '4': "} (27 August 1908 – 12 February 2001) was a film and television producer, director, and writer, born in
England to Australian parents.
</item 5>

<item 6>
Bill Barretta
William Paul "Bill" Barretta (born June 19, 1964) is an American puppeteer and producer who has been performing with The Muppets since 1991, when he puppeteered
the body of Sinclair family patriarch, Earl Sinclair on "Dinosaurs".

 He later developed several new characters on "Muppets Tonight", including Pepe the King Prawn, Johnny Fiama, Big Mean Carl and Bobo the Bear.

 Along with having his own Muppet characters, Barretta has taken over several of Jim Henson's roles, such as Dr. Teeth, Rowlf the Dog, Mahna Mahna and Swedish
Chef, and briefly took over Jerry Nelson's role of Lew Zealand.

 His film debut as a principal puppeteer was in 1996's "Muppet Treasure Island" as Clueless Morgan.

 In addition, Barretta has produced two of the Muppets' television films, "It's a Very Merry Muppet Christmas Movie" (2002) and "The Muppets' Wizard of Oz"
(2005).

 Barretta also provides additional voices on "Kim Possible".
</item 6>

<item 7>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 7>

<item 8>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 8>

<item 9>
Trey Parker
Randolph Severn "Trey" Parker III (born October 19, 1969) is an American actor, animator, writer, director, producer, singer, and songwriter.

 He is known for co-creating "South Park" (1997–present) along with his creative partner Matt Stone, as well as co-writing and co-directing the Tony
Award-winning musical "The Book of Mormon" (2011).

 Parker was interested in film and music as a child, and attended the University of Colorado, Boulder following high school, where he met Stone.

 The two collaborated on various short films, and starred in a feature-length musical, titled "Cannibal!

 The Musical" (1993).
</item 9>

<item 10>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 10>
=== END ITEM id=c644659bb9db9a35 ===

=== ITEM id=b737c7f6dcd174d4 ===
Gold sentences and facts:
Sentence 0: The Monastery of the Mother of God at the Spring (full name in Greek: , pr.
  Fact 0.0: The full Greek name of the Monastery of the Mother of God at the Spring is given, with the pronunciation abbreviation 'pr.'
Sentence 1:  "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an Eastern Orthodox sanctuary in Istanbul, Turkey.
  Fact 1.0: The Monastery of the Mother of God at the Spring has the name 'Moni tis Theotóku tis Pigis'.
  Fact 1.1: The Monastery of the Mother of God at the Spring has the Turkish name 'Balıklı Meryem Ana Rum Manastiri'.
  Fact 1.2: The Monastery of the Mother of God at the Spring is simply called Zoödochos Pege.
  Fact 1.3: Zoödochos Pege is Greek for 'Life-giving Spring' (Ζωοδόχος Πηγή).
  Fact 1.4: The Monastery of the Mother of God at the Spring is an Eastern Orthodox sanctuary.
  Fact 1.5: The Monastery of the Mother of God at the Spring is located in Istanbul, Turkey.
Sentence 2: Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.
  Fact 2.0: Hagia Irene is also called Hagia Eirene.
  Fact 2.1: Hagia Irene has the Ancient Greek name Ἁγία Εἰρήνη.
  Fact 2.2: A Byzantine form of the name is given for Hagia Irene, but the text of that form is cut off in the source.
  Fact 2.3: Hagia Irene's name means 'Holy Peace'.
  Fact 2.4: Hagia Irene has the Turkish name 'Aya İrini'.
  Fact 2.5: Hagia Irene is sometimes known also as Saint Irene.
  Fact 2.6: Hagia Irene is a Greek Eastern Orthodox church.
  Fact 2.7: Hagia Irene is located in the outer courtyard of Topkapı Palace.
  Fact 2.8: Topkapı Palace is in Istanbul, Turkey.

Retrieved items:
<item 1>
Church of St. Mary of the Spring (Istanbul)
The Monastery of the Mother of God at the Spring (full name in Greek: , pr.

 "Moni tis Theotóku tis Pigis"; Turkish name: "Balıklı Meryem Ana Rum Manastiri") or simply Zoödochos Pege (Greek: Ζωοδόχος Πηγή , "Life-giving Spring") is an
Eastern Orthodox sanctuary in Istanbul, Turkey.

 The present church, built in 1835, bears the same dedication as the shrine erected in this place between the end of the fifth and the beginning of the sixth
century.

 After several renovations, this building was destroyed in the first half of the fifteenth century by the Ottomans.

 The complex got its name from a nearby holy spring, reputed to have healing properties.

 For almost fifteen hundred years, this sanctuary has been one of the most important pilgrimage sites of Greek Orthodoxy.
</item 1>

<item 2>
St Mary de Lode Church, Gloucester
St Mary de Lode Church, Archdeacon Street, Gloucester GL1 2QT, is a Church of England church immediately outside the grounds of Gloucester Cathedral.

 It is believed by some to be on the site of the first Christian church in Britain.

 The church is in the Diocese of Gloucester and Grade I listed by English Heritage.

 It has also been known as "St. Mary Before the Gate of St. Peter", "St. Mary Broad Gate" and "St. Mary De Port".
</item 2>

<item 3>
St. Mary's Episcopal Church (Washington, D.C.)
St. Mary's Episcopal Church, also known as St. Mary's, Foggy Bottom or St. Mary's Chapel, is a historic Episcopal church located at 730 23rd Street, N.W. in the
Foggy Bottom neighborhood of Washington, D.C..

 On April 2, 1973, St. Mary's Episcopal Church was added to the National Register of Historic Places.
</item 3>

<item 4>
Hagia Irene
Hagia Irene or Hagia Eirene (Ancient Greek: Ἁγία Εἰρήνη , Byzantine ] , "Holy Peace", Turkish: "Aya İrini" ), sometimes known also as Saint Irene, is a Greek
Eastern Orthodox church located in the outer courtyard of Topkapı Palace in Istanbul, Turkey.

 It is one of the few churches in Istanbul that has not been converted into a mosque.

 The Hagia Irene today operates as a museum and concert hall.
</item 4>

<item 5>
St. Mary's Church (Swormville, New York)
St. Mary's Catholic Church is a church located on the corner of Transit (New York State Route 78) and Stahley roads in Swormville, New York, in the United
States.

 Originally known as St. Mary’s Church of the Assumption at Transit, the building was constructed of about 260,000 bricks.

 John Nepomucene Neumann (later to be named St. John Neumann) started to visit Swormville, celebrating mass in local homes, barns and fields.

 In 1839 he instructed that a small house would be made to hold mass.

 Bishop John Timon established the first catholic church (St. Mary's) of the town of Clarence, New York.

 In 1861, Father Michael Schinabeak made plans for a church to be constructed.

 Construction began in 1862 and finished in fall of 1865.

 The official opening of the Church was in January 1866.

 In 2010 a new Church was constructed behind the old St. Mary's.
</item 5>

<item 6>
St. Mary's Church and Rectory (Iowa City, Iowa)
St. Mary's Catholic Church, also known as St. Mary of the Visitation Church, is a parish church of the Diocese of Davenport which is located at 228 E. Jefferson
St. in Iowa City, Iowa, United States.

 The church building and rectory are listed together on the National Register of Historic Places in 1980.

 They were both included as contributing properties in the Jefferson Street Historic District in 2004.

 The parish's first rectory, which is now a private home, is also listed on the National Register as St. Mary's Rectory.

 It is located a few blocks to the east of the present church location at 610 E. Jefferson St.
</item 6>

<item 7>
Old Church of St George, Hill Street Dublin
The Old Church of St. George, commonly called "Little George's" in Hill Street (formerly Temple Street Lower) Parish of St. Mary, Dublin was built in 1668 by
the Eccles family for their workmen and also as a chapel-of-ease to the nearby St. Mary's Church.

 However, that St. Mary's Church was not St. Mary's Church, Dublin as its foundation stone was laid in 1700, and it was not St. Mary's Abbey, Dublin as it was
dissolved in 1539.

 Therefore St George's church, Hill Street, may have been a Chapel-of-Ease to St. Michan's Church in Church Street.

 The main body of the church, with the exception of the tower, was demolished in 1894.
</item 7>

<item 8>
Saint Michael the Archangel Church (Monroe, MI)
In 1918 the parish built the present building of St. Michael School which is now a part of Monroe Catholic Elementary Schools.

 The movement to establish the parish started in 1845 by 14 German immigrants who wanted to worship in their native language.

 The only nearby church was St. Mary of the Immaculate Conception right across the river where St. Michael stands today.

 St. Mary's only worshiped in French and wouldn't hold services in other languages like German and Irish.

 The German and Irish families were forced to move to the second floor of the rectory to have mass in their native languages.

 When the German families were able to acquire land they were eager to leave St. Mary's to start a parish of their own.

 They first purchased an empty lot Humphrey Street but the plan was soon abandoned.

 They then purchased the first mayor of Monroe's estate to build the church and it remains the current site of the church.
</item 8>

<item 9>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 9>

<item 10>
St. Mary's Church (Ballston Spa, New York)
St. Mary's Church is a Catholic parish located in Ballston Spa, New York.

 It is located within the Roman Catholic Diocese of Albany.

 Father Thomas J. Kelly is the current pastor.

 St. Mary's is the fourth oldest parish in the Diocese.

 St. Mary's of Ballston Spa is partnered with St. Mary's of Galway, after the pastor at St. Mary's of Galway died and no replacement was available.
</item 10>
=== END ITEM id=b737c7f6dcd174d4 ===

=== ITEM id=6c2c9f6f450ffb69 ===
Gold sentences and facts:
Sentence 0:  He appeared in 57 films between 1919 and 1932.
  Fact 0.0: The person described appeared in 57 films.
  Fact 0.1: The person described appeared in 57 films between 1919 and 1932.
Sentence 1:  He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
  Fact 1.0: The person described directed 54 films.
  Fact 1.1: The person described directed these films between 1920 and 1933.
  Fact 1.2: The person's directed films include one Harry Langdon short.
  Fact 1.3: The person's directed films include two of the Tay Garnett-penned comedies Stan Laurel made for Joe Rock.
  Fact 1.4: The person's directed films include fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
Sentence 2: John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.
  Fact 2.0: John Scott Biddle lived from 1925 to 2008.
  Fact 2.1: John Scott Biddle was a foremost yachting cinematographer.
  Fact 2.2: John Scott Biddle was a lecturer.
  Fact 2.3: John Scott Biddle established a film-making career.
  Fact 2.4: John Scott Biddle's film-making career spanned more than forty years.

Retrieved items:
<item 1>
John Biddle (Unitarian)
John Biddle or Bidle (born Wotton-under-Edge, Gloucestershire, England, 14 January 1615 – died 22 September 1662) was an influential English nontrinitarian, and
Unitarian.

 He is often called "the Father of English Unitarianism".
</item 1>

<item 2>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 2>

<item 3>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 3>

<item 4>
Charles John Biddle
Charles John Biddle (April 30, 1819 – September 28, 1873) was an American soldier, lawyer, congressman, and newspaper editor.
</item 4>

<item 5>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 5>

<item 6>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 6>

<item 7>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 7>

<item 8>
Kevin Brownlow
Kevin Brownlow (born 2 June 1938) is a British film historian, television documentary-maker, filmmaker, author, and film editor.

 Brownlow is best known for his work documenting the history of the silent era.

 Brownlow became interested in silent film at the age of eleven.

 This interest grew into a career spent documenting and restoring film.

 He has rescued many silent films and their history.

 His initiative in interviewing many largely forgotten, elderly film pioneers in the 1960s and 1970s preserved a legacy of early mass-entertainment cinema.

 Brownlow received an Academy Honorary Award at the 2nd Annual Governors Awards given by the Academy of Motion Picture Arts and Sciences on 13 November 2010.

 This was the first occasion on which an Academy Honorary Award was given to a film preservationist.
</item 8>

<item 9>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 9>

<item 10>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 10>
=== END ITEM id=6c2c9f6f450ffb69 ===

=== ITEM id=90822e34b3a97c2f ===
Gold sentences and facts:
Sentence 0:  It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).
  Fact 0.0: The company began in 1995.
  Fact 0.1: The company began as the merger of two take-and-bake pizza companies.
  Fact 0.2: One of the merged companies was Papa Aldo's Pizza.
  Fact 0.3: Papa Aldo's Pizza was founded in 1981.
  Fact 0.4: The other merged company was Murphy's Pizza.
  Fact 0.5: Murphy's Pizza was founded in 1981.
Sentence 1: Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.
  Fact 1.0: Gino's Pizza and Spaghetti is a restaurant chain.
  Fact 1.1: Gino's Pizza and Spaghetti has 40 locations.
  Fact 1.2: Most of Gino's Pizza and Spaghetti's locations are within the U.S. state of West Virginia.
Sentence 2:  The company was founded by Kenney Grant in 1961.
  Fact 2.0: The company was founded by Kenney Grant.
  Fact 2.1: The company was founded in 1961.

Retrieved items:
<item 1>
Nirula's
Nirula's is India's oldest fast food restaurant chain.

 Based in North India and most popular in NCR Delhi, it was Delhi's first fast food restaurant, opening in Connaught Place in 1977.

 Today it has over 70 outlets in NCR Delhi, Bihar, Haryana, Madhya Pradesh, Punjab, Rajasthan and Uttar Pradesh states, offering a “Desi” version of Western
fast food items.

 Nirula's success has led them to branch out into other ventures which include, ‘Potpourri’, an Indian cuisine, casual dining restaurant chain; ‘Nirula's 21’,
ice cream parlour chain, in addition to pastry shops and two hotels in Noida and Panipat.

 Recently Nirula's opened its first franchise in Patna, their first outlet in the entire east zone.
</item 1>

<item 2>
Papa Gino's
Papa Gino's, Inc. is a restaurant chain based in Dedham, Massachusetts specializing in American-style pizza along with pasta, subs, salads, and a variety of
appetizers.

 There are over 150 Papa Gino's locations in Connecticut, Maine, Massachusetts, New Hampshire, and Rhode Island.
</item 2>

<item 3>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 3>

<item 4>
Gino's East
Gino's East is a Chicago-based restaurant chain, notable for its deep-dish pizza (sometimes called Chicago-style pizza), and for its interior walls, which
patrons have covered in graffiti and etchings.

 The restaurant features deep-dish pizza baked in cast-iron pans, as well as sandwiches, soups and salads.
</item 4>

<item 5>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 5>

<item 6>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 6>

<item 7>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 7>

<item 8>
Papa Murphy's Park
Papa Murphy's Park (formerly known as the Cal Expo Multi-Use Sports Field Facility and Bonney Field) is a sports venue located on the grounds of Cal Expo in
Sacramento, California.

 The soccer-specific stadium has a capacity of 11,569 and includes a full-sized (120 x 80 yard) soccer field.

 Papa Murphy's Park is the current home of Sacramento Republic FC soccer team and former home of PRO Rugby team, Sacramento Express.
</item 8>

<item 9>
Gino's Pizza and Spaghetti
Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.

 The company was founded by Kenney Grant in 1961.

 Many locations are shared with Tudor's Biscuit World although the Gino's brand is exclusive to West Virginia.

 There is one located in Ohio, while there are stand alone Tudor's locations in eastern Kentucky, southern Ohio and southwest Virginia.

 Gino's serves pizza, spaghetti, sandwiches, and more.

 Company headquarters are located in Huntington, West Virginia and Nitro, West Virginia.
</item 9>

<item 10>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 10>
=== END ITEM id=90822e34b3a97c2f ===

=== ITEM id=469b059a6e1335cd ===
Gold sentences and facts:
Sentence 0:  There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.
  Fact 0.0: There is no optimal strategy involved in the game.
  Fact 0.1: Play in the game entirely rests upon a random drawing of the cards.
Sentence 1: Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.
  Fact 1.0: Diplomacy is a strategic board game.
  Fact 1.1: Diplomacy was created by Allan B. Calhamer.
  Fact 1.2: Diplomacy was created in 1954.
  Fact 1.3: Diplomacy was released commercially in 1959.
Sentence 2:  Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.
  Fact 2.0: Diplomacy's main distinctions from most board wargames include its negotiation phases.
  Fact 2.1: In Diplomacy's negotiation phases, players spend much of their time forming and betraying alliances with other players.
  Fact 2.2: In Diplomacy's negotiation phases, players spend much of their time forming beneficial strategies.
  Fact 2.3: Diplomacy's main distinctions from most board wargames include the absence of dice and other game elements that produce random effects.

Retrieved items:
<item 1>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 1>

<item 2>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 2>

<item 3>
Simultaneous action selection
Simultaneous action selection, or SAS, is a game mechanic that occurs when players of a game take action (such as moving their pieces) at the same time.

 An example of a game that uses this type of movement is the game Diplomacy.

 Typically, a "secret yet binding" method of committing to one's move is necessary, so that as players' moves are revealed and implemented, others do not change
their moves in light of the new information.

 Thus, in Diplomacy, players write down their moves and then reveal them simultaneously.

 Because no player gets the first move, this potentially arbitrary source of advantage is not present.

 It is also possible for simultaneous movement games to proceed relatively quickly, because players are acting at the same time, rather than waiting for their
turn.

 Simultaneous action selection is easily implemented in card games such as Apples to Apples in which players simply select cards and throw them face-down into
the center.
</item 3>

<item 4>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 4>

<item 5>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 5>

<item 6>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 6>

<item 7>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 7>

<item 8>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 8>

<item 9>
Diplomacy (game)
Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.

 Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players
and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.

 Set in Europe before the beginning of World War I, "Diplomacy" is played by two to seven players, each controlling the armed forces of a major European power
(or, with fewer players, multiple powers).

 Each player aims to move his or her few starting units and defeat those of others to win possession of a majority of strategic cities and provinces marked as
"supply centers" on the map; these supply centers allow players who control them to produce more units.
</item 9>

<item 10>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 10>
=== END ITEM id=469b059a6e1335cd ===

=== ITEM id=b8751efa33b4fc9c ===
Gold sentences and facts:
Sentence 0: The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.
  Fact 0.0: The Battle of Okinawa is also known as 沖縄戦 (Okinawa-sen in Hepburn romanization) and as 'Uchinaa ikusa' in Okinawan.
  Fact 0.1: The Battle of Okinawa was codenamed Operation Iceberg.
  Fact 0.2: The Battle of Okinawa was a major battle of the Pacific War.
  Fact 0.3: The Battle of Okinawa was fought on the island of Okinawa.
  Fact 0.4: The Battle of Okinawa was fought by United States Marine and Army forces against the Imperial Japanese Army.
  Fact 0.5: The Battle of Okinawa included the largest amphibious assault in the Pacific War during World War II.
  Fact 0.6: The largest amphibious assault in the Pacific War was the April 1, 1945 invasion of Okinawa itself.
Sentence 1: Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.
  Fact 1.0: Operation Diadem was also referred to as the Fourth Battle of Monte Cassino.
  Fact 1.1: In Canada, Operation Diadem was called the Battle of the Liri Valley.
  Fact 1.2: Operation Diadem was an offensive operation undertaken by the Allies of World War II.
  Fact 1.3: The Allies in Operation Diadem included the U.S. Fifth Army.
  Fact 1.4: The Allies in Operation Diadem included the British Eighth Army.
  Fact 1.5: Operation Diadem took place in May 1944.
  Fact 1.6: Operation Diadem was part of the Italian Campaign of World War II.

Retrieved items:
<item 1>
Operation Diadem
Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by
the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.

 "Diadem" was supported by air attacks called Operation Strangle.

 The opposing force was the German 10th Army.
</item 1>

<item 2>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 2>

<item 3>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 3>

<item 4>
Graph query row:
event: Operation Diadem
date: May - June 1944
kind: None
location: None
connections: [{'relationship': 'LOCATED_IN', 'entity': 'Winter Line'}, {'relationship': 'LOCATED_IN', 'entity': 'Anzio bridgehead'}, {'relationship': 'PART_OF',
'entity': 'Italian Campaign of World War II'}, {'relationship': 'PART_OF', 'entity': 'Allied breakthrough at Cassino'}, {'relationship': 'PART_OF', 'entity':
'breakout at Anzio'}, {'relationship': 'PART_OF', 'entity': 'capture of Rome'}]
</item 4>

<item 5>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 5>

<item 6>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 6>

<item 7>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 7>

<item 8>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 8>

<item 9>
Operation Kikusui
Operation Kikusui (菊水作戦 , Kikusui sakusen ) was a series of suicidal air attacks by Imperial Japanese forces during the Battle of Okinawa against Allied fleets
in the waters around Okinawa, as part of Operation Ten-Go.

 The name of the operation, "Kikusui", comes from the hata-jirushi of the samurai Kusunoki Masashige.
</item 9>

<item 10>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 10>
=== END ITEM id=b8751efa33b4fc9c ===

=== ITEM id=3bf24c11d9400d7f ===
Gold sentences and facts:
Sentence 0:  It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).
  Fact 0.0: The company began in 1995.
  Fact 0.1: The company began as the merger of two take-and-bake pizza companies.
  Fact 0.2: One of the merged companies was Papa Aldo's Pizza.
  Fact 0.3: Papa Aldo's Pizza was founded in 1981.
  Fact 0.4: The other merged company was Murphy's Pizza.
  Fact 0.5: Murphy's Pizza was founded in 1981.
Sentence 1: Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.
  Fact 1.0: Gino's Pizza and Spaghetti is a restaurant chain.
  Fact 1.1: Gino's Pizza and Spaghetti has 40 locations.
  Fact 1.2: Most of Gino's Pizza and Spaghetti's locations are within the U.S. state of West Virginia.
Sentence 2:  The company was founded by Kenney Grant in 1961.
  Fact 2.0: The company was founded by Kenney Grant.
  Fact 2.1: The company was founded in 1961.

Retrieved items:
<item 1>
Papa Gino's
Papa Gino's, Inc. is a restaurant chain based in Dedham, Massachusetts specializing in American-style pizza along with pasta, subs, salads, and a variety of
appetizers.

 There are over 150 Papa Gino's locations in Connecticut, Maine, Massachusetts, New Hampshire, and Rhode Island.
</item 1>

<item 2>
Lucky Luciano
Charles "Lucky" Luciano ( ; born Salvatore Lucania November 24, 1897 – January 26, 1962) was an Italian-American mobster and crime boss.

 Luciano is considered the father of modern organized crime in the United States for the establishment of the first Commission.

 He was also the first official boss of the modern Genovese crime family.

 He was, along with his associates, instrumental in the development of the National Crime Syndicate.
</item 2>

<item 3>
Graph query row:
organization: Gino's Pizza and Spaghetti
founding_date: 1961
founder_relationship: None
founder: None
</item 3>

<item 4>
Papa Murphy's
Papa Murphy's, a business based in Vancouver, Washington, United States, is a take-and-bake pizza company.

 It began in 1995 as the merger of two take-and-bake pizza companies: Papa Aldo's Pizza (founded in 1981) and Murphy's Pizza (founded in 1981).

 The company and its franchisees operate more than 1,300 outlets in the United States and Canada.

 Papa Murphy's is the fifth-largest pizza chain in the United States.
</item 4>

<item 5>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 5>

<item 6>
Gino's Pizza and Spaghetti
Gino's Pizza and Spaghetti is a restaurant chain with 40 locations, most of them within the U.S. state of West Virginia.

 The company was founded by Kenney Grant in 1961.

 Many locations are shared with Tudor's Biscuit World although the Gino's brand is exclusive to West Virginia.

 There is one located in Ohio, while there are stand alone Tudor's locations in eastern Kentucky, southern Ohio and southwest Virginia.

 Gino's serves pizza, spaghetti, sandwiches, and more.

 Company headquarters are located in Huntington, West Virginia and Nitro, West Virginia.
</item 6>

<item 7>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 7>

<item 8>
Prem Ganapathy
Prem Ganapathy is an Indian entrepreneur and businessman.

 He is the founder of the restaurant chain Dosa plaza.

 Starting with meager investment, he expanded Dosa plaza into a restaurant chain with 45 outlets in India, New Zealand, Oman, and UAE.
</item 8>

<item 9>
Graph query row:
organization: Papa Murphy's
founding_date: 1995
founder_relationship: None
founder: None
</item 9>

<item 10>
Gino's East
Gino's East is a Chicago-based restaurant chain, notable for its deep-dish pizza (sometimes called Chicago-style pizza), and for its interior walls, which
patrons have covered in graffiti and etchings.

 The restaurant features deep-dish pizza baked in cast-iron pans, as well as sandwiches, soups and salads.
</item 10>
=== END ITEM id=3bf24c11d9400d7f ===

=== ITEM id=ad8f7e64400764ef ===
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
Samantha Stosur career statistics
This is a list of the main career statistics of professional Australian tennis player, Samantha Stosur.

 To date, Stosur has won eight WTA singles titles including one Grand Slam singles title at the 2011 US Open.

 She was also the runner-up at the 2010 French Open and a semi-finalist at the 2010 and 2011 WTA Tour Championships.

 Stosur has also enjoyed a successful doubles career, in which she has held the World No. 1 ranking and won twenty-four WTA doubles titles including two grand
slam women's doubles titles at the 2005 US Open and 2006 French Open and two year-ending championships at the 2005 and 2006 WTA Tour Championships.
</item 1>

<item 2>
Carly Gullickson
Carly Gullickson (born November 26, 1986) is a former American professional tennis player.
</item 2>

<item 3>
Francesca Schiavone
Francesca Schiavone (] ; born 23 June 1980 in Milan) is an Italian tennis player who turned professional in 1998.

 She won the 2010 French Open singles title, becoming the first Italian woman to win a Grand Slam event in singles.

 She was also runner-up at the 2011 French Open.

 Her career high ranking is world No. 4, achieved on 31 January 2011.

 To date, Schiavone is the last one handed-backhand player to win a Grand Slam title on the women's tour.
</item 3>

<item 4>
1999 Italian Open – Women's Doubles
The 1999 Italian Open – Doubles was the doubles event of the fifty-fifth edition of the tennis tournament played at Rome, Italy, the most prestigious tennis
tournament in Southern Europe.

 It was the fifth WTA Tier I tournament of the year, and part of the European claycourt season.

 Virginia Ruano Pascual and Paola Suárez were the defending champions, but lost to Martina Hingis and Anna Kournikova in the quarterfinals.
</item 4>

<item 5>
2005 Challenge Bell – Doubles
Carly Gullickson and María Emilia Salerni were the defending champions, but Gullickson decided not to participate this year.

 Salerni partnered with Marion Bartoli, but withdrew from their semifinal match against Līga Dekmeijere and Ashley Harkleroad.
</item 5>

<item 6>
2010 US Open – Mixed Doubles
Carly Gullickson and Travis Parrott were the defending champions, but lost in the first round to Gisela Dulko and Pablo Cuevas 2–6, 4–6.
</item 6>

<item 7>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 7>

<item 8>
2006 Qatar Ladies Open – Doubles
Alicia Molik and Francesca Schiavone were the defending champions, but Molik did not compete this year.

 Schiavone teamed up with Květa Peschke and lost in querterfinals to Elena Likhovtseva and Vera Zvonareva.
</item 8>

<item 9>
2007 Kremlin Cup – Women's Doubles
Květa Peschke and Francesca Schiavone were the defending champions, but Schiavone chose not to participate, and only Peschke competed that year.
</item 9>

<item 10>
2015 Apia International Sydney – Women's Doubles
Tímea Babos and Lucie Šafářová were the defending champions, but chose not to participate together.

 Babos played alongside Kristina Mladenovic, but lost in the semifinals to Raquel Kops-Jones and Abigail Spears.

 Šafářová teamed up with Belinda Bencic, but lost in the first round to Kimiko Date-Krumm and Karolína Plíšková.

<br>
</item 10>
=== END ITEM id=ad8f7e64400764ef ===

=== ITEM id=34b19a4664e04495 ===
Gold sentences and facts:
Sentence 0: The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna (Tenerife, Canary Islands, Spain).
  Fact 0.0: The Church of the Guanche People's Spanish name is 'Iglesia del Pueblo Guanche'.
  Fact 0.1: The Church of the Guanche People is a religious organisation.
  Fact 0.2: The Church of the Guanche People was founded in 2001.
  Fact 0.3: The Church of the Guanche People was founded in the city of San Cristobal de La Laguna.
  Fact 0.4: San Cristobal de La Laguna is in Tenerife, Canary Islands, Spain.
Sentence 1:  Tenerife is the largest and most populous island of Macaronesia.
  Fact 1.0: Tenerife is the largest island of Macaronesia.
  Fact 1.1: Tenerife is the most populous island of Macaronesia.

Retrieved items:
<item 1>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 1>

<item 2>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 2>

<item 3>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
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
Adenocarpus viscosus
Adenocarpus viscosus is a shrubby species of flowering plant in the legume family Fabaceae, subfamily Faboideae.

 It is endemic to the Canary Islands where it is known locally as Codeso del Pico.

 It can be found above 1800 m on two of the islands, La Palma in Caldera de Tabouriente and Tenerife where it is a dominant shrub in Teide National Park and
occurs in parts of Corona Forestal Nature Park and Reserva Especial de las Palomas.
</item 6>

<item 7>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 7>

<item 8>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 8>

<item 9>
Chiguata District
Cherubs also decorate the circular cornice, along with four saints in full-length reliefs that occupy the "pechinas" of the columns that sustain the temple's
dome, which is considered one of the most original works of the colony and now visited by occasional tourists.
</item 9>

<item 10>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 10>
=== END ITEM id=34b19a4664e04495 ===

=== ITEM id=1fc7cf815c725d76 ===
Gold sentences and facts:
Sentence 0: McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.
  Fact 0.0: McFarland, USA is also known as McFarland.
  Fact 0.1: McFarland, USA is a 2015 American sports drama film.
  Fact 0.2: McFarland, USA was directed by Niki Caro.
  Fact 0.3: McFarland, USA was produced by Mark Ciardi.
  Fact 0.4: McFarland, USA was produced by Gordon Gray.
  Fact 0.5: McFarland, USA was written by Christopher Cleveland.
  Fact 0.6: McFarland, USA was written by Bettina Gilois.
  Fact 0.7: McFarland, USA was written by Grant Thompson.
  Fact 0.8: McFarland, USA has music composed by Antônio Pinto.
Sentence 1: The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.
  Fact 1.0: The Black Hole is a 1979 American science fiction film.
  Fact 1.1: The Black Hole was directed by Gary Nelson.
  Fact 1.2: The Black Hole was produced by Walt Disney Productions.

Retrieved items:
<item 1>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 1>

<item 2>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 2>

<item 3>
The Black Hole
The Black Hole is a 1979 American science fiction film directed by Gary Nelson and produced by Walt Disney Productions.

 The film stars Maximilian Schell, Robert Forster, Joseph Bottoms, Yvette Mimieux, Anthony Perkins, and Ernest Borgnine, while the voices of the main robot
characters are provided by Roddy McDowall and Slim Pickens (both unbilled).

 The music for the film was composed by John Barry.

 It was the first film from Walt Disney Productions to receive a PG rating.

 The film was released on December 18, 1979 in the United Kingdom and on December 21, 1979 in the United States.
</item 3>

<item 4>
Steve McQueen (director)
Steven Rodney "Steve" McQueen (born 9 October 1969) is an English film director, producer, screenwriter, and video artist.

 For his 2013 film, "12 Years a Slave", a historical drama adaptation of an 1853 slave narrative memoir, he won an Academy Award, BAFTA Award for Best Film, and
Golden Globe Award for Best Motion Picture – Drama, as a producer, and he also received the award for Best Director from the New York Film Critics Circle.

 McQueen is the first black filmmaker to win an Academy Award for Best Picture.

 McQueen is known for his collaborations with actor Michael Fassbender, who has starred in all three of McQueen's feature films as of 2014.

 McQueen's other feature films are "Hunger" (2008), a historical drama about the 1981 Irish hunger strike, and "Shame" (2011), a drama about an executive
struggling with sex addiction.
</item 4>

<item 5>
A broad film, television, documentary, and industry collection spans silent-era cinema through 2010, covering action, comedy, drama, horror, musicals, noir, and
more. Highlights include *The Razor’s Edge*, *The Little Foxes*, *Die Hard*, *Unbreakable*, *Signs*, *The Straight Story*, *West Side Story*, and *Rogue One*.
Documentaries address music, war, politics, disability, and culture. Profiles feature Bruce Willis, David
</item 5>

<item 6>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 6>

<item 7>
The entries cover diverse films and careers: crime drama *Deadfall* and its *Arsenal* follow-up; *Rogue One*’s rebels stealing Death Star plans; Mel Gibson,
Nicolas Cage, James Garner and Tom Cruise filmographies; Everest disaster drama *Everest*; integration-themed sports films *Remember the Titans* and *McFarland,
USA*; acclaimed *True Grit* and *The Hurt Locker*; espionage film *The Ipcress File*; Vietnam
</item 7>

<item 8>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 8>

<item 9>
The entries cover films across action, comedy, romance, horror, and drama. Bruce Willis stars in titles including *Last Man Standing* (a credited *Yojimbo*
remake), *Unbreakable*, *Signs*, and *Die Hard with a Vengeance*. Shyamalan’s *Signs* explores faith and alien encounters; *Unbreakable* launches a superhero
trilogy. Other highlights include *The Whole Nine Yards*, *Striking Distance*, *Mercury Rising
</item 9>

<item 10>
Full Metal Jacket
Full Metal Jacket is a 1987 British-American war film directed and produced by Stanley Kubrick.

 The screenplay by Kubrick, Michael Herr, and Gustav Hasford was based on Hasford's novel "The Short-Timers" (1979).

 Its storyline follows a platoon of U.S. Marines through their training, primarily focusing on two privates, Joker and Pyle, who struggle to get through camp
under their foul-mouthed drill instructor, Gunnery Sergeant Hartman, and the experiences of two of the platoon's Marines in the Tet Offensive during the Vietnam
War.

 The film's title refers to the full metal jacket bullet used by soldiers.

 The film was released in the United States on June 26, 1987.
</item 10>
=== END ITEM id=1fc7cf815c725d76 ===
