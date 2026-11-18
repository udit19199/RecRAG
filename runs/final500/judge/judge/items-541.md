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

=== ITEM id=bb650a25f1a82976 ===
Gold sentences and facts:
Sentence 0: Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.
  Fact 0.0: Henderson Point is a census-designated place (CDP).
  Fact 0.1: Henderson Point is in Harrison County, Mississippi, United States.
Sentence 1:  Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
  Fact 1.0: Gulfport is a city adjoining Biloxi.
  Fact 1.1: Biloxi is a county seat of Harrison County.

Retrieved items:
<item 1>
Jasper County Community Unit School District 1
Jasper County Community Unit School District 1 is a unified school district based in Jasper County's county seat of Newton, Illinois; it is the only school
district in the county and is, consequently, the main educational body in all of Jasper County, although it serves portions of Effingham County and Cumberland
County as well.

 This school district is composed of six schools in total; four elementary schools, one junior high school, and one high school.

 There is also a prekindergarten program run at the high school of the district should parents wish to enroll their children early.

 Willow Hill Elementary School, which is located in the village of its namesake, serves only kindergarteners; its proximity to the county seat and central
position in the county allows its students to dawn from all parts of the county and still have easy access to the elementary school they will attend.

 The principal of Willow Hill is Dave Parker.

 Grove Elementary School is located in Island Grove, Illinois, the highest point of elevation in the county.

 Grove Elementary School educates students from kindergarten to grade six, and it runs a prekindergarten program as well.

 Craig Carr is the principal of this school.
</item 1>

<item 2>
Paulding, Mississippi
Paulding is an unincorporated community in and one of the two county seats of Jasper County, Mississippi.

 It is the only unincorporated county seat in Mississippi.

 Settled in 1833, it was named by United States settlers in honor of Revolutionary War hero John Paulding.

 After its citizens refused to contribute to a new railroad, the community was bypassed in favor of Bay Springs, Mississippi, which was designated a railroad
stop to the west and the second county seat.

 It attracted more development and industry.
</item 2>

<item 3>
Jasper County Community Unit School District 1
Craig Carr is the principal of this school.

 Ste. Marie Elementary School is located in the southern Jasper County village of Ste. Marie, and serves students in grades one through six.

 The principal of Ste. Marie is David Parker, the principal of Willow Hill Elementary School.

 Newton Elementary School is located in the county seat of Newton, and serves most of west Jasper County's first through sixth graders under principal Travis
Wyatt.

 The latter three elementary schools feed into Jasper County Junior High School and are taught in the facility during seventh and eighth grade while being
supervised by Newton Elementary School principal Travis Wyatt before graduating into Newton Community High School.

 Students in grades nine through twelve spend their last leg of precollegiate education at this school; their principal is Ruth Kerner.

 The district superintendent is Ron Alburtus, and the district's mascot is the eagle.
</item 3>

<item 4>
Oldham County, KY (2010: 60,316; seat La Grange), named for William Oldham, became fully wet in 2015 after limited restaurant sales began in 2005. Casey County
(15,955; Liberty), formed 1806, is dry, Appalachian, wholly in the Knobs, and hosts an Apple Festival. Virginia’s Kentucky County (1776–c.1780) had its seat at
Harrodstown. Piney Woods School,
</item 4>

<item 5>
Henderson, Texas
Henderson is a city in Rusk County, northeast Texas, United States.

 The population was 13,712 at the 2010 census.

 It is the county seat of Rusk County.

 Henderson is named for James Pinckney Henderson, the first governor of Texas.
</item 5>

<item 6>
Biloxi, Mississippi
Biloxi ( ) is a city in Harrison County, Mississippi, United States.

 The 2010 United States Census recorded the population as 44,054, and in 2016 the estimated population was 45,975.

 Along with the adjoining city of Gulfport, Biloxi is a county seat of Harrison County.
</item 6>

<item 7>
Hendersonville, North Carolina
Hendersonville is a city in Henderson County, North Carolina United States.

 This city is 22 miles south of Asheville.

 It is the county seat of Henderson County.

 Like the county, the city is named for 19th-century North Carolina Supreme Court Chief Justice Leonard Henderson.
</item 7>

<item 8>
Piney Woods Country Life School
The Piney Woods Country Life School (or The Piney Woods School) is a co-educational independent historically African-American boarding school for grades 9-12 in
Piney Woods, unincorporated Rankin County, Mississippi.

 It is 21 mi south of Jackson.

 It is one of four remaining historically African-American boarding schools in the United States.

 It is currently the largest African-American boarding school, as well as being the second oldest continually operating African-American boarding school.
</item 8>

<item 9>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 9>

<item 10>
Henderson Point, Mississippi
Henderson Point is a census-designated place (CDP) in Harrison County, Mississippi, United States.

 It is part of the Gulfport–Biloxi Metropolitan Statistical Area.

 The population was 170 at the 2010 census.
</item 10>
=== END ITEM id=bb650a25f1a82976 ===

=== ITEM id=e992e9942b0541ea ===
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
Grant Nicholas
Grantley Jonathan "Grant" Nicholas (born 12 November 1967) is a Welsh musician, best known as the lead singer and lead guitarist of the rock band Feeder.
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
Sweet Nothing
"Sweet Nothing" is a song by Scottish recording artist and record producer Calvin Harris from his third studio album, "18 Months" (2012).

 It features vocals from Florence and the Machine singer Florence Welch.

 Harris previously worked with the band in a remix of their single, "Spectrum" (2012).

 While recording "18 Months", Harris expressed interest in working with Welch.

 He mentioned that it was not easy and he had to chase her, due to schedule conflicts.

 Welch accepted his invitation and the two recorded "Sweet Nothing".

 The track premiered on "The Chris Moyles Show" on BBC Radio 1 on 28 August 2012 and was later released as the fifth single from the album on 12 October 2012.
</item 5>

<item 6>
Nelson (band)
Nelson is an American rock band founded by singer/songwriters Matthew and Gunnar Nelson (twin sons of Ricky Nelson and Kristin Nelson).

 The band achieved success during the early 1990s with their double platinum debut album "After the Rain", which featured the number-one hit "(Can't Live
Without Your) Love and Affection".
</item 6>

<item 7>
Ceremonials Tour
The Ceremonials Tour was the second concert tour by English indie rock band Florence and the Machine.

 The tour included performances at Music festivals because it is lead singer Florence Welch's favourite way to perform live.

 Welch had originally planned to spend over one year touring for "Ceremonials" (their previous tour lasted almost three and a half years) before announcing that
the December 2012 dates would be the final performances of the tour.

 On 25 February 2013 it was announced that Florence and the Machine would be playing at the Coke Live Music Festival in Poland on 10 August 2013, and on 26
March 2013 it was announced that the band would be playing at Chime for Change's "The Sound of Change Live" concert at London's Twickenham Stadium alongside
Beyoncé, Ellie Goulding and Haim, amongst others.
</item 7>

<item 8>
Florence Welch
Florence Leontine Mary Welch (born 28 August 1986) is an English musician, singer and songwriter, best known as the vocalist of indie rock band Florence and the
Machine.
</item 8>

<item 9>
Romeo Nelson
Iromeio "Romeo" Nelson (March 12, 1902 – May 17, 1974) was an American boogie woogie pianist whose recordings from 1929 are regarded as some of the finest, and
certainly the fastest, boogie woogie showpieces on record.
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
=== END ITEM id=e992e9942b0541ea ===

=== ITEM id=f13390d87cd4df81 ===
Gold sentences and facts:
Sentence 0: The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy Schaal and Henry Gibson.
  Fact 0.0: The 'Burbs is a 1989 American film.
  Fact 0.1: The 'Burbs is a comedy thriller film.
  Fact 0.2: The 'Burbs was directed by Joe Dante.
  Fact 0.3: The 'Burbs stars Tom Hanks.
  Fact 0.4: The 'Burbs stars Bruce Dern.
  Fact 0.5: The 'Burbs stars Carrie Fisher.
  Fact 0.6: The 'Burbs stars Rick Ducommun.
  Fact 0.7: The 'Burbs stars Corey Feldman.
  Fact 0.8: The 'Burbs stars Wendy Schaal.
  Fact 0.9: The 'Burbs stars Henry Gibson.
Sentence 1: Richard "Rick" Ducommun (July 3, 1952 – June 12, 2015) was a Canadian actor, comedian and writer who appeared in films and television.
  Fact 1.0: Rick Ducommun was born on July 3, 1952.
  Fact 1.1: Rick Ducommun died on June 12, 2015.
  Fact 1.2: Rick Ducommun was a Canadian actor.
  Fact 1.3: Rick Ducommun was a Canadian comedian.
  Fact 1.4: Rick Ducommun was a Canadian writer.
  Fact 1.5: Rick Ducommun appeared in films.
  Fact 1.6: Rick Ducommun appeared in television.

Retrieved items:
<item 1>
Carrie Fisher
Carrie Frances Fisher (October 21, 1956 – December 27, 2016) was an American actress, writer, and humorist.

 Fisher was known for playing Princess Leia in the "Star Wars" films; a role for which she was nominated for three Saturn Awards.

 Her other film roles included "Shampoo" (1975), "The Blues Brothers" (1980), "Hannah and Her Sisters" (1986), "The 'Burbs" (1989), "When Harry Met Sally..."
(1989), "Soapdish" (1991) and "The Women" (2008 film).

 She was nominated twice for the Primetime Emmy Award for Outstanding Guest Actress in a Comedy Series for her performances on the television series "30 Rock"
and "Catastrophe".

 She was posthumously made a Disney Legend in 2017.
</item 1>

<item 2>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 2>

<item 3>
Neon City
Neon City (also known as Anno 2053 in Italy, Neonski Grad in Serbia) is a 1991 Canadian post-apocalyptic science fiction film directed by Monte Markham and
written by Jeff Begun and Ann Lewis Hamilton, under the pseudonym Buck Finch.

 The cast includes Michael Ironside, Vanity, Lyle Alzado, Valerie Wildman, Nick Klar, Juliet Landau, Richard Sanders and Markham.
</item 3>

<item 4>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
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
Araby Lockhart
Araby Lockhart (born December 4, 1926) is a Canadian stage and television actress, best known for her performances in the films "Capote" and "Police Academy"
and her stage performances as a member of Hart House Theatre and the Straw Hat Players.

 Lockhart has also served as President of the Actors' Fund of Canada.
</item 6>

<item 7>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 7>

<item 8>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 8>

<item 9>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 9>

<item 10>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
</item 10>
=== END ITEM id=f13390d87cd4df81 ===

=== ITEM id=cd32de5cf34fd638 ===
Gold sentences and facts:
Sentence 0: Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.
  Fact 0.0: Lee Dong-hae was born on October 15, 1986.
  Fact 0.1: Lee Dong-hae's name in Hangul is 이동해.
  Fact 0.2: Lee Dong-hae's name in Hanja is 李東海.
  Fact 0.3: Lee Dong-hae is a South Korean singer.
  Fact 0.4: Lee Dong-hae is a South Korean songwriter.
  Fact 0.5: Lee Dong-hae is a South Korean actor.
Sentence 1: Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.
  Fact 1.0: Costadinos Contostavlos was born on 11 June 1987.
  Fact 1.1: Costadinos Contostavlos is better known by his stage name Dappy.
  Fact 1.2: Costadinos Contostavlos is an English singer.
  Fact 1.3: Costadinos Contostavlos is an English songwriter.
  Fact 1.4: Costadinos Contostavlos is an English rapper.
  Fact 1.5: Costadinos Contostavlos is an English actor.

Retrieved items:
<item 1>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 1>

<item 2>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 2>

<item 3>
Eunhyuk
Lee Hyuk-jae (born April 4, 1986), better known by his stage name Eunhyuk, is a South Korean singer-songwriter and actor.

 He is a member of the South Korean boy group Super Junior and their subgroups, Super Junior-T and Super Junior-H.

 In 2011, he joined Super Junior's Mandopop subgroup, Super Junior-M and the duo Donghae & Eunhyuk; active in China and Japan respectively.
</item 3>

<item 4>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 4>

<item 5>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 5>

<item 6>
Dappy
Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.

 He is best known for being the lead singer of the Camden-based grime trio N-Dubz, with his cousin Tulisa, and Fazer.

 He was known for his slightly unusual dress sense and his love for what he describes as "eye-catching headwear".

 His trademark was wearing a selection of woolly chullos (sometimes referred to as "Dappy hats"), turning up one or both of the ear-flaps of the hat.
</item 6>

<item 7>
Yang Yo-seob
Yang Yo-seob (born January 5, 1990), more commonly known as Yoseob, is a South Korean singer and musical actor.

 He is the main vocalist of the boy group Highlight.
</item 7>

<item 8>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 8>

<item 9>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 9>

<item 10>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 10>
=== END ITEM id=cd32de5cf34fd638 ===

=== ITEM id=aff98081dd374ff1 ===
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
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 1>

<item 2>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 2>

<item 3>
Robert Oliveri
Robert Dane Oliveri (born April 28, 1978) is a former American actor.

 He is best known for his portrayal of Nick Szalinski in the 1989 Disney film, "Honey, I Shrunk the Kids", for which he was nominated for a Young Artist Award
and a Saturn Award.

 He reprised the role in the 1992 sequel, "Honey, I Blew Up the Kid" and the 3D film, "Honey, I Shrunk the Audience".

 He is also known as Kevin, Winona Ryder's little brother, in Tim Burton's "Edward Scissorhands".
</item 3>

<item 4>
Marty Belafsky
Marty Belafsky (born September 19, 1975) is an American actor/comedian born in Los Angeles, California.

 He began acting professionally at age 13 and was soon cast as Louis Plumb on the short-lived NBC series "Hull High".

 Shortly thereafter, Belafsky landed the role of Crutchy in the Disney musical film, "Newsies".

 He continued acting through his teens, making appearances in such television shows as "The Wonder Years", "Great Scott" and "Step By Step" and the film
"Wrestling Ernest Hemingway".

 Belafsky also voiced Kent Swanson in the video game series "Dead Rising" released in 2006.
</item 4>

<item 5>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
</item 5>

<item 6>
The Cool Ones
The Cool Ones (aka "Cool, Baby Cool") is a 1967 film starring Roddy McDowall and directed by Gene Nelson.

 Mrs. Miller performs in a cameo role, and the film features cameo performances by bands the Leaves and the Bantams, as well as a cameo appearance by Glen
Campbell.
</item 6>

<item 7>
Bring It On: In It to Win It
Bring It On: In It to Win It is a 2007 teen film directed by Steve Rash and starring Ashley Benson, Cassie Scerbo and Michael Copon.
</item 7>

<item 8>
Steve Rash
Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and
"".
</item 8>

<item 9>
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 9>

<item 10>
Can't Buy Me Love (film)
Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in
Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

 The film was directed by Steve Rash and takes its title from a Beatles song with the same title.
</item 10>
=== END ITEM id=aff98081dd374ff1 ===

=== ITEM id=25a7ecd0fc24535b ===
Gold sentences and facts:
Sentence 0: 3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.
  Fact 0.0: 3:10 to Yuma is a 2007 American Western film.
  Fact 0.1: 3:10 to Yuma was directed by James Mangold.
  Fact 0.2: 3:10 to Yuma was produced by Cathy Konrad.
  Fact 0.3: Russell Crowe and Christian Bale star in the lead roles of 3:10 to Yuma.
  Fact 0.4: Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman have supporting performances in 3:10 to Yuma.
Sentence 1: Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and Wat in "A Knight's Tale".
  Fact 1.0: Alan Wray Tudyk was born on March 16, 1971.
  Fact 1.1: Alan Tudyk is an American actor.
  Fact 1.2: Alan Tudyk is an American voice actor.
  Fact 1.3: Alan Tudyk is known for playing Hoban "Wash" Washburne in the space western television series "Firefly" and in the movie "Serenity".
  Fact 1.4: Alan Tudyk is known for playing Alpha in the science fiction TV series "Dollhouse".
  Fact 1.5: Alan Tudyk is known for playing Tucker McGee in "Tucker & Dale vs. Evil".
  Fact 1.6: Alan Tudyk is known for playing Steve the Pirate in "" (the title is missing from the source sentence).
  Fact 1.7: Alan Tudyk is known for playing Wat in "A Knight's Tale".

Retrieved items:
<item 1>
The Appaloosa
The Appaloosa (also known as Southwest to Sonora) is a 1966 American Western film Technicolor (set in the 1870s) from Universal Pictures starring Marlon Brando,
Anjanette Comer and John Saxon, who was nominated for a Golden Globe for Best Supporting Actor for his portrayal of a Mexican bandit.

 The film was directed by Sidney J. Furie, and shot in Wrightwood, Antelope Valley, and Lake Los Angeles, California, St. George, Utah, and Colorado City,
Arizona.
</item 1>

<item 2>
Henry Murdoch
Henry Murdoch (17 September, 1920 - 24 April, 1987), born as George Henry Murdock, was an Australian aboriginal actor and stockman who appeared in Australian
films of the 1940s and 1950s.

 He was working as stockman in Rockhampton when discovered by Ralph Smart, who was helping make "The Overlanders" (1946).

 The film's director, Harry Watt, later claimed Murdoch and fellow aboriginal actor Clyde Combo "proved to be first-class actors and were exceedingly quick
witted and intelligent.

 They certainly disproved the conventional idea that the Australian aboriginal is an animalistic caveman."
</item 2>

<item 3>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 3>

<item 4>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 4>

<item 5>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 5>

<item 6>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 6>

<item 7>
Alan Tudyk
Alan Wray Tudyk ( ; born March 16, 1971) is an American actor and voice actor known for his roles as Hoban "Wash" Washburne in the space western television
series "Firefly" and movie "Serenity", Alpha in the science fiction TV series "Dollhouse", Tucker McGee in "Tucker & Dale vs. Evil", Steve the Pirate in "", and
Wat in "A Knight's Tale".

 He also co-starred on the ABC sitcom "Suburgatory".

 Tudyk had a supporting role in the film "42" as baseball manager Ben Chapman and voiced roles in every Walt Disney Animation Studios film to date starting with
"Wreck-It Ralph".

 He has also featured in the web series "Con Man" and the films "" and "Rogue One: A Star Wars Story".
</item 7>

<item 8>
Mel Gibson filmography
Mel Gibson, AO, is an American actor, director, producer and screenwriter who made his acting debut on the Australian television drama series "The Sullivans"
(1976– 1983).

 While a student at the National Institute of Dramatic Art in Sydney, he was given an uncredited role in "I Never Promised You a Rose Garden" and subsequently
appeared as a leading actor in the micro budget surf drama "Summer City" (both in 1977).

 Gibson rose to prominence during the Australian New Wave cinema movement in the early 1980s, having appeared in his breakthrough role in George Miller's
dystopian action film "Mad Max" (1979), portraying the eponymous hero.

 He reprised the role in its sequels, "Mad Max 2" (1981) and "Mad Max Beyond Thunderdome" (1985).

 He appeared in Peter Weir's war drama "Gallipoli" (1981) and the romantic drama "The Year of Living Dangerously" (1982).
</item 8>

<item 9>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 9>

<item 10>
3:10 to Yuma (2007 film)
3:10 to Yuma is a 2007 American Western film directed by James Mangold and produced by Cathy Konrad, and starring Russell Crowe and Christian Bale in the lead
roles, with supporting performances by Peter Fonda, Gretchen Mol, Ben Foster, Dallas Roberts, Alan Tudyk, Vinessa Shaw, and Logan Lerman.

 It is about a drought-impoverished rancher (Bale) who takes on the dangerous job of taking a notorious outlaw (Crowe) to justice.

 It is a remake of the , making it the second adaptation of Elmore Leonard's short story "Three-Ten to Yuma".

 Filming took place in various locations in New Mexico.

 "3:10 to Yuma" opened September 7, 2007, in the United States and received positive reviews from critics.
</item 10>
=== END ITEM id=25a7ecd0fc24535b ===

=== ITEM id=f6378ed981c506b4 ===
Gold sentences and facts:
Sentence 0:  The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI Studios in London, which was soon renamed Abbey Road Studios.
  Fact 0.0: The title and cover are an homage to the Beatles album Abbey Road.
  Fact 0.1: 926 East McLemore Avenue is the address of the Stax Studios in Memphis.
  Fact 0.2: Abbey Road was the address of EMI Studios in London.
  Fact 0.3: EMI Studios in London was soon renamed Abbey Road Studios.
Sentence 1:  This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
  Fact 1.0: This road is best known for the Abbey Road Studios.
  Fact 1.1: This road is best known for the 1969 album Abbey Road by The Beatles.

Retrieved items:
<item 1>
List of roads named after Mahatma Gandhi
Mahatma Gandhi Road which is also used in its abbreviated form as M. G. Road, named after Mahatma Gandhi.

 M. G. Road is one of the most frequently used road names in India, Agra having two of them.

 Several other Indian cities have a road by this name.

 In 2010, attempts were made to rename Hillcroft Avenue in Houston, Texas to Mahatma Gandhi Avenue but the proponents were unable to secure signatures from 75%
of the property owners.

 As a result, the group settled for renaming the areas around Hillcroft Avenue to Mahatma Gandhi district.

 In Kolkata, Central Road was renamed Harrison Road in 1892 after Sir Henry Leland Harrison, then the chairman of Calcutta Corporation.

 It has recently been renamed yet again to Mahatma Gandhi Road.

 In many cities in South India, Mahatma Gandhi Road is popularly known by other names.

 In Chennai, it is known as Nungambakkam High Road.

 In Coimbatore, Mahatma Gandhi Road is popularly known by Avarampalayam Road.
</item 1>

<item 2>
Westminster St Margaret and St John
St Margaret was an ancient parish in the City and Liberty of Westminster and the county of Middlesex.

 It included the core of modern Westminster, including the Palace of Westminster and the area around, but not including Westminster Abbey.

 In 1727 it was divided into St Margaret's and St John's, to coincide with the building of the Church of St John the Evangelist, constructed by the Commission
for Building Fifty New Churches in Smith Square to meet the demands of the growing population, but there continued to be a single vestry for the parishes of St
Margaret and St John.

 This was reformed in 1855 by the Metropolis Management Act, and the two parishes formed the Westminster District until 1887.

 In 1889 St Margaret and St John became part of the County of London.

 The vestry was abolished in 1900, to be replaced by Westminster City Council, but St Margaret and St John continued to have a nominal existence until 1922.
</item 2>

<item 3>
Westminster
Westminster ( ) is an area of central London within the City of Westminster, part of the West End, on the north bank of the River Thames.

 Westminster's concentration of visitor attractions and historic landmarks, one of the highest in London, includes the Palace of Westminster, Buckingham Palace,
Westminster Abbey and Westminster Cathedral.
</item 3>

<item 4>
Abbey Road, London
Abbey Road is a thoroughfare in the borough of Camden and the City of Westminster in London, running roughly northwest to southeast through St. John's Wood,
near Lord's Cricket Ground.

 It is part of the B507 road.

 This road is best known for the Abbey Road Studios and the 1969 album, "Abbey Road", by The Beatles.
</item 4>

<item 5>
City of Westminster Magistrates' Court
The City of Westminster Magistrates' Court was a magistrates' court located at 70 Horseferry Road, in the City of Westminster, London.

 It was originally called Horseferry Road Magistrates' Court, after the road in which it was sited.

 However, it was renamed in July 2006 following the closure of Bow Street Magistrates' Court.

 It served as the court where the Chief Magistrate of England and Wales sat, and all extradition and terrorism-related cases passed through the court.

 The court closed permanently on 22 September 2011, and was replaced on 27 September 2011 with Westminster Magistrates' Court, built on the site of Marylebone
Magistrates' Court at 181 Marylebone Road.
</item 5>

<item 6>
Westminster tube station
Westminster is a London Underground station in the City of Westminster.

 It is served by the Circle, District and Jubilee lines.

 On the Circle and District lines, the station is between St. James's Park and Embankment, and on the Jubilee line it is between Green Park and Waterloo.

 It is in Travelcard Zone 1.

 The station is located at the corner of Bridge Street and Victoria Embankment and is close to the Houses of Parliament, Westminster Abbey, Parliament Square,
Whitehall, Westminster Bridge, and the London Eye.

 Also close by are Downing Street, the Cenotaph, Westminster Millennium Pier, the Treasury, the Foreign and Commonwealth Office, and the Supreme Court.
</item 6>

<item 7>
Stax Museum of American Soul Music
The Stax Museum of American Soul Music is a museum located in Memphis, Tennessee, at 926 East McLemore Avenue, the former location of Stax Records.

 It is operated by Soulsville USA, which also operates the adjacent Stax Music Academy.
</item 7>

<item 8>
Temple Bar, London
Temple Bar was the principal ceremonial entrance to the City of London on its western side from the City of Westminster.

 It is situated on the historic royal ceremonial route from the Tower of London to the Palace of Westminster, the two chief residences of the mediaeval English
monarchs, and from the Palace of Westminster to St Paul's Cathedral.

 The road east of Temple Bar and within the City is Fleet Street, the road to the west, in Westminster, is The Strand.

 At Temple Bar the Corporation of the City of London formerly erected a barrier to regulate trade into the City.

 The 19th century Royal Courts of Justice are located next to it on its north side, having been moved from Westminster Hall.

 To its south is the Temple Church and the Inner Temple and Middle Temple Inns of Court.

 As the most important entrance to the City of London from Westminster, it was formerly long the custom for the monarch to halt at Temple Bar before entering
the City of London, in order for the Lord Mayor to offer the Corporation's pearl-encrusted Sword of State as a token of loyalty.
</item 8>

<item 9>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 9>

<item 10>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 10>
=== END ITEM id=f6378ed981c506b4 ===

=== ITEM id=bbb8aa3d4be99a7a ===
Gold sentences and facts:
Sentence 0:  He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and the Highest Distinction of the Argentine Air Force.
  Fact 0.0: He was awarded the highest national military decoration, the Argentine Nation to the Heroic Valour in Combat Cross.
  Fact 0.1: He was awarded the Argentine Congressional Medal.
  Fact 0.2: He was awarded the Highest Distinction of the Argentine Air Force.
Sentence 1:  In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
  Fact 1.0: In 2010, the entity referred to here had 14,600 military personnel.
  Fact 1.1: In 2010, the entity referred to here had 6,900 civilian personnel.

Retrieved items:
<item 1>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 1>

<item 2>
Pablo Carballo
Commodore Pablo Marcos Rafael Carballo (b. 11 December 1947 Buenos Aires) is a retired member of the Argentine Air Force - the "Fuerza Aérea Argentina" (FAA) -
who fought in the 1982 Falklands War ( Spanish: "Guerra de las Malvinas" ) where he participated in actions that led to the sinking of three Royal Navy ships.

 He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and
the Highest Distinction of the Argentine Air Force.
</item 2>

<item 3>
Cuerpo de Fuerzas Especiales
The Mexican Cuerpo de Fuerzas Especiales (Special Forces Corps) is a special forces unit of the Mexican Army.

 Formerly the "GAFE" (Grupo Aeromóvil de Fuerzas Especiales | Special-Forces Airmobile Group), the SF Corps has six battalions; one is the "Fuerza especial de
reaccion", a quick-response unit, and one is assigned to the Paratroopers Rifle Brigade; the motto of the SF Corps is "Todo por México" (Everything for Mexico).

 Within the SF Corps, there are regular, intermediate, and veteran -service troops.

 The regular-service soldiers usually operate as light infantry.

 The intermediate-service soldiers (lieutenants and captains) usually are instructors.

 The veteran-service soldiers of the Grupo Aeromóvil de Fuerzas Especiales del Alto Mando (GAFE High Command) handle Black-Ops missions.

 Also known as the COIFE, the Special Forces Corps of the Mexican Army is equivalent to the U.S. Army Special Forces.
</item 3>

<item 4>
South America air forces maneuvers
The South American air forces performs several joint aerial combat training exercises.

 Among the more important are Cruzex ( Portuguese: "Exercício Cruzeiro do Sul" ) which is hosted by the Brazilian Air Force, Salitre ( English: Saltpeter )
hosted by the Chilean Air Force and Ceibo ( English: Erythrina crista-galli flower ) hosted by the Argentine Air Force.

 The goal is to train together in order to respond to a crisis or integrate into United Nations peacekeeping operations as a unified team; the United States Air
Force and the Royal Canadian Air Force have also participated in some of these exercises in recent years.
</item 4>

<item 5>
The passage profiles special forces worldwide: the Philippine Army’s airborne regiment trains with U.S. Green Berets; Britain’s SAS conducts reconnaissance,
counterterrorism, direct action, and hostage rescue; Singapore’s SOF serves in the SOTF; Japan’s WWII Giretsu paratroopers targeted Allied air raids; Chile’s
Iquique commandos belong to an armored brigade; Jordan’s 1963-founded forces perform intelligence, rescue, counterterrorism, and precision strikes; South
African Jan B
</item 5>

<item 6>
Argentine Air Force
The Argentine Air Force (Spanish: Fuerza Aérea Argentina , or simply FAA) is the national aviation branch of the Armed Forces of the Argentine Republic.

 In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
</item 6>

<item 7>
Argentine Air Force Mobile Field Hospital
The Argentine Air Force Mobile Field Hospital (Spanish: "Hospital Reubicable de la Fuerza Aérea" ) is a field hospital operated by the Argentine Air Force.

 It is one of three health centers of its kind worldwide.
</item 7>

<item 8>
Teodoro Waldner
Teodoro Guillermo Waldner (born 1927) is a retired Argentine Air Force commander who served as Air Force Chief of the General Staff from 1983 to 1985.
</item 8>

<item 9>
Servicio de Inteligencia de la Fuerza Aérea
Servicio de Inteligencia de la Fuerza Aérea ("Air Force Intelligence Service", SIFA) is the intelligence agency of the Argentine Air Force.

 It is part of J-2.

 Its director is Commodore García.
</item 9>

<item 10>
Hector Fautario
Hector Luis Fautario was a retired Argentine Air Force general.

 He was the General Commander of the Argentine Air Force from 1973 to 1975.

 Shortly prior to the 1976 Argentine coup d'état, he was the government's only remaining loyal officer.

 He had drawn harsh criticism towards himself from the Army and Navy due to his vehement opposition to their repressive plans, and for his refusal to mobilize
the Air Force against the guerrillas' strongholds in the north.

 He was General Jorge Rafael Videla's last obstacle on the way to power.
</item 10>
=== END ITEM id=bbb8aa3d4be99a7a ===

=== ITEM id=40b38f42785c9517 ===
Gold sentences and facts:
Sentence 0: ", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)
  Fact 0.0: This person did voice work for My Life As A Teenage Robot.
  Fact 0.1: This person did voice work for The Replacements.
  Fact 0.2: This person did the voice of Knotty in a 1992 work whose title is blank in the sentence.
  Fact 0.3: This person did additional voices in the film Fantasia 2000 (1999).
  Fact 0.4: This person did additional voices in the film Sing (2016).
Sentence 1:  It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.
  Fact 1.0: The film was directed by Garth Jennings.
  Fact 1.1: The film was written by Garth Jennings.
  Fact 1.2: The film was co-directed by Christophe Lourdelet.
  Fact 1.3: The film starred the voices of Matthew McConaughey.
  Fact 1.4: The film starred the voices of Reese Witherspoon.
  Fact 1.5: The film starred the voices of Seth MacFarlane.
  Fact 1.6: The film starred the voices of Scarlett Johansson.
  Fact 1.7: The film starred the voices of John C. Reilly.
  Fact 1.8: The film starred the voices of Taron Egerton.
  Fact 1.9: The film starred the voices of Tori Kelly.

Retrieved items:
<item 1>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 1>

<item 2>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 2>

<item 3>
Small Apartments
Small Apartments is a 2012 American comedy film directed by Jonas Åkerlund.

 It tells the story of Franklin Franklin, played by Matt Lucas, who by mistake kills his landlord, Mr. Olivetti, played by Peter Stormare.

 The cast co-stars Dolph Lundgren, Johnny Knoxville, James Caan, Billy Crystal, Juno Temple, Rebel Wilson, Saffron Burrows and Amanda Plummer.

 The screenplay was written by Chris Millis and adapted from his own novella.

 The film premiered at the South by Southwest Film Festival on March 10, 2012.
</item 3>

<item 4>
Winnie the Pooh (2011 film)
Winnie the Pooh is a 2011 American animated buddy musical comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 51st Disney animated feature film.

 Inspired by A. A. Milne's stories of the same name, the film is part of Disney's "Winnie the Pooh" franchise, the fifth theatrical "Winnie the Pooh" film
released, and Walt Disney Animation Studios' second adaptation of "Winnie-the-Pooh" stories.

 Jim Cummings reprises his vocal roles as Winnie the Pooh and Tigger, while series newcomers Travis Oates, Tom Kenny, Craig Ferguson, Bud Luckey, and Kristen
Anderson-Lopez provide the voices of Piglet, Rabbit, Owl, Eeyore, and Kanga, respectively.

 In the film, the aforementioned residents of the Hundred Acre Wood embark on a quest to save Christopher Robin from an imaginary culprit while Pooh deals with
a hunger for honey.
</item 4>

<item 5>
Are We Done Yet?
Are We Done Yet?

 is a 2007 American family comedy film starring Ice Cube.

 The film is a remake of the Cary Grant comedy "Mr. Blandings Builds His Dream House" and a sequel to the 2005 comedy "Are We There Yet?

" The film was directed by Steve Carr from a screenplay by Hank Nelken.

 It was produced by Revolution Studios and RKO Pictures and distributed by Columbia Pictures.

 The film was shot on location in Tsawwassen, British Columbia, Canada, but is set in Newberg, Oregon, United States.
</item 5>

<item 6>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 6>

<item 7>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 7>

<item 8>
♯JusticeForFlint
Initiated by "Creed" director Ryan Coogler and "Selma" director Ava DuVernay, the free public event was held at the Whiting Auditorium in Flint, Michigan.

 Hosted by comedian Hannibal Buress, it featured singers Janelle Monáe and Ledisi, as well as actor-activists Jesse Williams and Jussie Smollett, amongst
others.

 Surprising star guest was "King of Motown" Stevie Wonder, who teamed up with Monáe for "Hell You Talkin' Bout", a protest song she had written for the people
of Flint.
</item 8>

<item 9>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 9>

<item 10>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 10>
=== END ITEM id=40b38f42785c9517 ===

=== ITEM id=e84fcd76e22605df ===
Gold sentences and facts:
Sentence 0: Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.
  Fact 0.0: Ravi Sethi was born in 1947.
  Fact 0.1: Ravi Sethi is an Indian computer scientist.
  Fact 0.2: Ravi Sethi retired from Bell Labs.
  Fact 0.3: Ravi Sethi is president of Avaya Labs Research.
Sentence 1: Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company, owned by Finnish company Nokia.
  Fact 1.0: Nokia Bell Labs was formerly named AT&T Bell Laboratories.
  Fact 1.1: Nokia Bell Labs was formerly named Bell Telephone Laboratories.
  Fact 1.2: Nokia Bell Labs was formerly named Bell Labs.
  Fact 1.3: Nokia Bell Labs is an American research and scientific development company.
  Fact 1.4: Nokia Bell Labs is owned by the Finnish company Nokia.
Sentence 2:  Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
  Fact 2.0: The headquarters of Nokia Bell Labs are located in Murray Hill, New Jersey.
  Fact 2.1: Nokia Bell Labs has other laboratories around the rest of the United States.
  Fact 2.2: Nokia Bell Labs has laboratories in other countries.

Retrieved items:
<item 1>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 1>

<item 2>
Center for Climate Systems Research (CCSR)
The Center for Climate Systems Research is a key Earth Institute center with over 25 scientists and staff researching issues involving the interplay between
Earth's climate and society.

 The center's motto is "Columbia's Gateway to NASA and Beyond", as it has a special relationship with the NASA Goddard Institute for Space Studies (GISS).

 It is a "Gateway to NASA", because the center is co-located with GISS in Columbia University's Armstrong Hall above Tom's Restaurant in New York City.

 In this historic location, Columbia and NASA scientists work together closely to understand the Earth's climate and how changes can impact society both in the
United States and around the world.

 Importantly, CCSR scientists also work with many other partners – governmental, private sector, and non-governmental organizations – to answer these
challenging scientific and socioeconomic questions (i.e., "and Beyond").

 The director of CCSR is Dr. Michael J. Puma.
</item 2>

<item 3>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 3>

<item 4>
The profiles span medicine, science, technology, education, and philanthropy. James P. Comer’s school-development program has reached over 600 schools; Pamela
Davis leads Case Western Reserve’s medical school. Ravi Sethi co-authored the “Dragon Book,” while David Bailey researches mathematics and computing. Other
notable figures include Nobel laureate Peter Doherty; former acting U.S. Surgeon General Steven Galson; engineer Edward Keonjian, an early microelectronics
pioneer; and researchers in
</item 4>

<item 5>
Umesh Varshney
Umesh Varshney (born 1957) is an Indian molecular biologist, academician and the head of the "Prof.Umesh Varshney's Lab" at the Indian Institute of Science,
Bengaluru.

 He is a J. C. Bose National Fellow of the Department of Science and Technology and is known for his studies on protein synthesis and DNA repair in "Escherichia
coli" and Mycobacterium tuberculosis.

 An elected fellow of the Indian Academy of Sciences, Indian National Science Academy and the National Academy of Sciences (India), he is also a recipient of
the National Bioscience Award for Career Development of the Government of India.

 The Council of Scientific and Industrial Research, the apex agency of the Government of India for scientific research, awarded him the Shanti Swarup Bhatnagar
Prize for Science and Technology, one of the highest Indian science awards, in 2001, and then in 2014 with the G. N. Ramachandran Gold Medal for Excellence in
Biological Sciences & Technology for his contributions to biological sciences.
</item 5>

<item 6>
Franz Josef Och
Franz Josef Och (November 2, 1971) is a German computer scientist.

 He is currently working at Grail (an Illumina company) as Head of Data Science.

 Prior to this he was Chief Data Scientist at Human Longevity Inc., and earlier worked for Google as a Distinguished Research Scientist and head of machine
translation based at Google's Mountain View, California, headquarters south of San Francisco.
</item 6>

<item 7>
Ravi Sethi
Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.

 He also serves as a member of the National Science Foundation's Computer and Information Science and Engineering (CISE) Advisory Committee.

 He is best known as one of three authors of the classic computer science textbook "", also known as the "Dragon Book".
</item 7>

<item 8>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 8>

<item 9>
Sethi–Ullman algorithm
In computer science, the Sethi–Ullman algorithm is an algorithm named after Ravi Sethi and Jeffrey D. Ullman, its inventors, for translating abstract syntax
trees into machine code that uses as few registers as possible.
</item 9>

<item 10>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 10>
=== END ITEM id=e84fcd76e22605df ===

=== ITEM id=e7737cc819a1f166 ===
Gold sentences and facts:
Sentence 0: Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
  Fact 0.0: Rolf Martin Theodor Dahlgren was born on 7 July 1932.
  Fact 0.1: Rolf Martin Theodor Dahlgren died on 14 February 1987.
  Fact 0.2: Rolf Martin Theodor Dahlgren was a Swedish-Danish botanist.
  Fact 0.3: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen.
  Fact 0.4: Rolf Martin Theodor Dahlgren was a professor at the University of Copenhagen from 1973 until his death.
Sentence 1:  The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.
  Fact 1.0: The university has 23,473 undergraduate students.
  Fact 1.1: The university has 17,398 postgraduate students.
  Fact 1.2: The university has 2,968 doctoral students.
  Fact 1.3: The university has over 9,000 employees.

Retrieved items:
<item 1>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 1>

<item 2>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 2>

<item 3>
Ebba Witt-Brattström
Ebba Witt-Brattström (born 1953) is a Swedish scholar in comparative literature.

 She is Professor of Literature and head of department at Södertörn University outside Stockholm, and a well-known feminist.

 In the 1970s she was a member of the feminist organisation Grupp 8, and in 2005 she was one of the founders of the feminist political organisation and party
Feministiskt Initiativ, although she later distanced herself from the organisation and criticized its alleged strong left-wing tendencies.
</item 3>

<item 4>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 4>

<item 5>
Gertrud Dahlgren
Gertrud Dahlgren (1931–2009) was a Swedish botanist.

 She was born at Klippan in Scania, and after graduating from university at Helsingborg went to the University of Lund for graduate studies.

 There she obtained her M.Sc.

 in chemistry and biology.

 She was married to fellow botanist Rolf Dahlgren (1932–1987), who was killed in a car accident.

 The couple had three children, Susanna, Helena and Fredrik.

 She died in December 2009 at the age of 78.
</item 5>

<item 6>
John D. Odegard School of Aerospace Sciences
The John D. Odegard School of Aerospace Sciences (UND Aerospace) is a multidisciplinary college within the University of North Dakota (UND) in Grand Forks,
North Dakota.

 The school was formed in 1968.

 The majority of the school's fleet of over 120 aircraft is based at nearby Grand Forks International Airport and is the largest fleet of civilian flight
training aircraft in North America.

 UND Aerospace also operates flight training centers in Crookston, Minnesota, and Phoenix, Arizona.

 Today, the school has many aerospace-related programs including commercial aviation (fixed wing and rotorcraft), air traffic control, airport management, Space
Studies, Computer Science, Atmospheric Sciences, and Earth System Science & Policy.

 Currently, the school has over 500 faculty and 1,900 students making it the second largest of UND’s degree-granting colleges.

 The present dean of the school is Dr. Paul Lindseth.
</item 6>

<item 7>
Rolf M. Zinkernagel
Rolf Martin Zinkernagel {'1': ", '2': ", '3': ", '4': "} (born January 6, 1944 in Riehen, Basel-Stadt, Switzerland) is Professor of Experimental Immunology at
the University of Zurich.

 He was awarded the Nobel Prize in Physiology or Medicine in 1996 for the discovery of how the immune system recognizes virus-infected cells.
</item 7>

<item 8>
Rolf Dahlgren
Rolf Martin Theodor Dahlgren (7 July 1932 – 14 February 1987) was a Swedish-Danish botanist, professor at the University of Copenhagen from 1973 to his death.
</item 8>

<item 9>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 9>

<item 10>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 10>
=== END ITEM id=e7737cc819a1f166 ===
