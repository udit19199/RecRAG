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

=== ITEM id=ce9282f8c4c75415 ===
Gold sentences and facts:
Sentence 0: Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.
  Fact 0.0: Hayley Nichole Williams was born on December 27, 1988.
  Fact 0.1: Hayley Nichole Williams is an American singer.
  Fact 0.2: Hayley Nichole Williams is an American songwriter.
  Fact 0.3: Hayley Nichole Williams is an American musician.
Sentence 1:  The band consists of Hayley Williams, Zac Farro and Taylor York.
  Fact 1.0: The band consists of Hayley Williams.
  Fact 1.1: The band consists of Zac Farro.
  Fact 1.2: The band consists of Taylor York.
Sentence 2: Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American heavy metal bands Lord Tracy and Pantera.
  Fact 2.0: Terrence Lee Glaze was born on November 29, 1964, in Columbus.
  Fact 2.1: Terrence Lee Glaze is known as Terry Lee.
  Fact 2.2: Terry Lee is a Texas-raised singer.
  Fact 2.3: Terry Lee is a Texas-raised guitarist.
  Fact 2.4: Terry Lee is most known for his work in the American heavy metal band Lord Tracy.
  Fact 2.5: Terry Lee is most known for his work in the American heavy metal band Pantera.

Retrieved items:
<item 1>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 1>

<item 2>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 2>

<item 3>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 3>

<item 4>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 4>

<item 5>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 5>

<item 6>
Hayley Williams (ice hockey)
Hayley Williams (born June 3, 1990) is an American professional ice hockey player for the Brampton Thunder of the Canadian Women's Hockey League (CWHL).

 Williams has previously played for the Buffalo Beauts in the National Women's Hockey League (NWHL).
</item 6>

<item 7>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 7>

<item 8>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 8>

<item 9>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=ce9282f8c4c75415 ===

=== ITEM id=30f6f4f3b3bfd938 ===
Gold sentences and facts:
Sentence 0:  It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
  Fact 0.0: This song is most known through the 1989 single by Kathy Mattea.
  Fact 0.1: The 1989 single by Kathy Mattea was released in conjunction with her album Willow in the Wind.
  Fact 0.2: This song was first recorded and released on the 1987 Don Williams album Traces.
  Fact 0.3: This song was also released in 1988 by Clark's husband on his album Old Friends.
Sentence 1: This is a detailed discography for American country music artist Don Williams.
  Fact 1.0: The discography is detailed.
  Fact 1.1: The discography is for American country music artist Don Williams.

Retrieved items:
<item 1>
Hjärtats trakt
Hjärtats trakt was released on 20 April 1993 and is a compilation album from Swedish pop artist Per Gessle.

 The album includes songs taken from Gessle's solo albums "Per Gessle" and "Scener".

 It was also released under license by the British record company Pickwick and it is the only Gessle album released by another label than EMI.

 In 1998 the album was re-released by the Dutch record company Disky Communications B.V. which is a subsidiary of EMI.
</item 1>

<item 2>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 2>

<item 3>
Come On (Earl King song)
"Come On" (often called "Let the Good Times Roll") is a song written by New Orleans rhythm and blues artist Earl King.

 He first recorded the song as "Darling Honey Angel Child" in 1960 for the Ace Records subsidiary Rex.

 Later that year, he recorded it as a two-part song for Imperial Records using some new lyrics.

 Retitled "Come On", it was released in 1960 with "Come On - Part I” as the A-side backed with “Come On - Part II” (Imperial 5713).
</item 3>

<item 4>
Leave Them Boys Alone
"Leave Them Boys Alone" is a song recorded by American country music artist Hank Williams Jr. with Waylon Jennings and Ernest Tubb.

 It was released in May 1983 as the second single from Williams' album "Strong Stuff".

 The song reached number 6 on the "Billboard" Hot Country Singles chart.

 It was written by Williams, Dean Dillon, Gary Stewart and Tanya Tucker.

 The song is notable for its combination of two singers associated with the outlaw movement with a country legend from the honky tonk days and golden age of the
Grand Ole Opry.

 Outlaw singers like Williams and Jennings saw themselves as taking country music back to its raw, honky tonk roots, and recording an up tempo song with Tubb
(who would never have received radio airplay in the late 1970s and early 80's) and reaching #6 was a slap in the face to the proponents of the country pop
sound.

 The lyrics of the song, much like Williams' "Family Tradition" echo the sentiment that the outlaw singers and their current escapades were predated by the hard
living honky tonkers of the 1950s such as Hank Williams, Sr.
</item 4>

<item 5>
Night Is Fallin' in My Heart
"Night Is Fallin' in My Heart" is a song written by Dennis Linde.

 The song was first recorded by Country music artist J.P. Pennington for his 1991 album, "Whatever It Takes".

 It was later recorded and released as a single in October 1994 by American country music group Diamond Rio.

 It was the second single released from their third album, "Love a Little Stronger".

 It peaked at number 9 in the United States, and number 6 in Canada.
</item 5>

<item 6>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 6>

<item 7>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 7>

<item 8>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 8>

<item 9>
Every Beat of My Heart (song)
"Every Beat of My Heart" is a rhythm and blues song by Johnny Otis.

 It was first recorded in 1954 by his group, The Royals (later to be known as The Midnighters).

 In 1961 Gladys Knight & the Pips recorded the song for their debut single on the Vee-Jay label.

 Credited to The Pips, it was the first of eleven releases by the group to make it to number one on the R&B/soul chart.

 It was also the group's first top ten on the "Billboard" Hot 100.

 "Every Beat of My Heart" was first recorded for the Huntom label, who later sold the master to Vee-Jay.

 At the time of the song's release, The Pips were on the Fury label where they re-recorded the song without piano.

 In an unusual occurrence, the Fury recording of the song also made the top twenty on the R&B Sides chart and also made the Hot 100.
</item 9>

<item 10>
Hate to See Your Heart Break
"Hate to See Your Heart Break" is a song by American rock band Paramore, recorded for their 2013 self-titled fourth album "Paramore".

 It was re-recorded to feature vocals by Joy Williams (formerly of The Civil Wars) for the 2014 deluxe edition of the album, the first time Paramore has
collaborated with another artist or group on a studio recording.
</item 10>
=== END ITEM id=30f6f4f3b3bfd938 ===

=== ITEM id=283ee71e6eb211b4 ===
Gold sentences and facts:
Sentence 0: Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.
  Fact 0.0: Sleepless in Seattle is a 1993 film.
  Fact 0.1: Sleepless in Seattle is an American film.
  Fact 0.2: Sleepless in Seattle is a romantic comedy-drama film.
  Fact 0.3: Sleepless in Seattle was directed by Nora Ephron.
  Fact 0.4: Sleepless in Seattle was co-written by Nora Ephron.
  Fact 0.5: Sleepless in Seattle is based on a story by Jeff Arch.
Sentence 1:  It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber, and Rita Wilson.
  Fact 1.0: Sleepless in Seattle stars Tom Hanks.
  Fact 1.1: Sleepless in Seattle stars Meg Ryan.
  Fact 1.2: Sleepless in Seattle has a supporting cast featuring Bill Pullman.
  Fact 1.3: Sleepless in Seattle has a supporting cast featuring Ross Malinger.
  Fact 1.4: Sleepless in Seattle has a supporting cast featuring Rob Reiner.
  Fact 1.5: Sleepless in Seattle has a supporting cast featuring Rosie O'Donnell.
  Fact 1.6: Sleepless in Seattle has a supporting cast featuring Gaby Hoffmann.
  Fact 1.7: Sleepless in Seattle has a supporting cast featuring Victor Garber.
  Fact 1.8: Sleepless in Seattle has a supporting cast featuring Rita Wilson.
Sentence 2: Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.
  Fact 2.0: Roseann O'Donnell was born on March 21, 1962.
  Fact 2.1: Roseann O'Donnell is an American comedian.
  Fact 2.2: Roseann O'Donnell is an American actress.
  Fact 2.3: Roseann O'Donnell is an American author.
  Fact 2.4: Roseann O'Donnell is an American television personality.

Retrieved items:
<item 1>
Dennis Fimple
Dennis Clarke Fimple (November 11, 1940 – August 23, 2002) was an American character actor.

 He appeared in a variety of TV shows including "Here Come the Brides", "Petticoat Junction", "Matt Houston", "M*A*S*H", "Centennial", "Simon & Simon", "Sledge
Hammer!

", "Knight Rider", "Quantum Leap" and "ER".

 He also had roles in films such as "Truck Stop Women" (1974), "The Apple Dumpling Gang" (1975), "Mackintosh and T.J." (1975), "Stay Hungry" (1976), "King Kong"
(1976), "The Shadow of Chikara" (1977), "Goin' South" (1978), "The Wild Women of Chastity Gulch" (1982) and "Maverick" (1994), and shared the lead in
"Bootleggers" (1974) and "Creature from Black Lake" (1976).
</item 1>

<item 2>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 2>

<item 3>
Matthew Perry
Matthew Langford Perry (born August 19, 1969) is a Canadian-American actor, best known for his role as Chandler Bing on the long-running NBC television sitcom
"Friends", as well as his portrayal of Ron Clark in the 2006 television movie "The Ron Clark Story".
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
In My Sleep
In My Sleep is a 2010 suspense thriller film written, directed and produced by film director Allen Wolf.

 It stars Philip Winchester, Lacey Chabert, Tim Draxl, Abigail Spencer and Kelly Overton.

 It also features Kirsten Vangsness from Criminal Minds in a cameo role and Tony Hale from Veep in a cameo as well.

 The story is about a massage therapist with chronic insomnia who fears he may have murdered a good friend while sleepwalking.
</item 5>

<item 6>
Carrie Fisher
Carrie Frances Fisher (October 21, 1956 – December 27, 2016) was an American actress, writer, and humorist.

 Fisher was known for playing Princess Leia in the "Star Wars" films; a role for which she was nominated for three Saturn Awards.

 Her other film roles included "Shampoo" (1975), "The Blues Brothers" (1980), "Hannah and Her Sisters" (1986), "The 'Burbs" (1989), "When Harry Met Sally..."
(1989), "Soapdish" (1991) and "The Women" (2008 film).

 She was nominated twice for the Primetime Emmy Award for Outstanding Guest Actress in a Comedy Series for her performances on the television series "30 Rock"
and "Catastrophe".

 She was posthumously made a Disney Legend in 2017.
</item 6>

<item 7>
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 7>

<item 8>
Rosie O'Donnell
Roseann O'Donnell (born March 21, 1962) is an American comedian, actress, author, and television personality.

 She has been a magazine editor and continues to be a celebrity blogger, a lesbian rights activist, a television producer, and a collaborative partner in the
LGBT family vacation company, R Family Vacations.
</item 8>

<item 9>
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
</item 9>

<item 10>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 10>
=== END ITEM id=283ee71e6eb211b4 ===

=== ITEM id=93577ebdbd8a567a ===
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
Howard Rosenman
Howard Rosenman (born February 1, 1945), also known as Zvi Howard Rosenman, is an American producer and motion picture executive.

 He specializes in producing romantic comedy films and documentary films.

 Some of his most popular productions include "Father of the Bride" (1991) starring Steve Martin and Diane Keaton, Joss Whedon's "Buffy the Vampire Slayer"
(1992) and "The Family Man" (2000) starring Nicolas Cage.

 Rosenman's documentary film "" won the Peabody Award and the 1990 Academy Award for Best Documentary Feature; his film "The Celluloid Closet" also won the
Peabody Award.
</item 1>

<item 2>
Can't Buy Me Love (film)
Can't Buy Me Love is a 1987 American teen romantic comedy feature film starring Patrick Dempsey and Amanda Peterson in a story about a nerd at a high school in
Tucson, Arizona who gives a cheerleader $1,000 to pretend to be his girlfriend for a month.

 The film was directed by Steve Rash and takes its title from a Beatles song with the same title.
</item 2>

<item 3>
Just Friends
Just Friends is a 2005 American romantic comedy film directed by Roger Kumble, written by Adam 'Tex' Davis and starring Ryan Reynolds, Amy Smart, Anna Faris,
Chris Klein and Christopher Marquette.

 The plot focuses on a formerly overweight high school nerd (Reynolds) who attempts to free himself from the friend zone after reconnecting with his lifelong
crush and best friend (Smart) while visiting his hometown for Christmas.
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
Bring It On: All or Nothing
Bring It On: All or Nothing (previously known as Bring It On Yet Again) is a 2006 cheerleading comedy film directed by Steve Rash and starring Hayden Panettiere
and Solange Knowles.

 It is the third installment in the "Bring it On" series of films that revolves around high school cheerleading.

 The film was released direct-to-DVD on August 8, 2006.
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
Steve Rash
Steve Rash is an American film director and producer best known for directing such films as "The Buddy Holly Story", "Can't Buy Me Love", "Queens Logic", "" and
"".
</item 7>

<item 8>
American Pie Presents: Band Camp
American Pie Presents: Band Camp (also known as American Pie: Band Camp) is a 2005 direct-to-DVD sex comedy film.

 It is the first installment in the "American Pie Presents" series and the fourth installment in the "American Pie" franchise.

 It was directed by Steve Rash and written by Brad Riddell.

 The film centers on Matt Stifler, younger brother of Steve Stifler.

 In this film, Matt is sent to the infamous band camp, where he is compelled to change his ways.
</item 8>

<item 9>
Jimmy Tatro
James Richard Tatro (born February 16, 1992) is an American actor, comedian, writer and YouTube personality.

 He is the creator and star of the YouTube channel "LifeAccordingToJimmy", which has about 2.7 million subscribers and over 440 million video views.

 Tatro writes, produces and directs each of his video sketches with his friend, Christian A. Pierce.

 Tatro has appeared in "Divergent", "Grown Ups 2", "22 Jump Street" and "", and will appear in the film "Camp Manna" alongside Gary Busey, set to be released in
2017.

 He is also starring in the true-crime satire, "American Vandal", in which he plays Dylan Maxwell, the accused vandal.

 He will also appear in the movie, Action No. 1, a film about a group of people who attempt to steal the first comic book with Superman off the actor Nicholas
Cage.
</item 9>

<item 10>
Bring It On: In It to Win It
Bring It On: In It to Win It is a 2007 teen film directed by Steve Rash and starring Ashley Benson, Cassie Scerbo and Michael Copon.
</item 10>
=== END ITEM id=93577ebdbd8a567a ===

=== ITEM id=73fb398e5226f5db ===
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
Vektroid
Ramona Andra Xavier (born August 19, 1992, known by her primary alias Vektroid) is an American electronic musician from Portland, Oregon.

 She has released music under her primary Vektroid alias, as well as others, such as dstnt, Laserdisc Visions, New Dreams Ltd., Macintosh Plus, Virtual
Information Desk, and PrismCorp Virtual Enterprises.

 Xavier played a prominent role in the popularization of the vaporwave subgenre with the release of her only studio album under her Macintosh Plus alias,
"Floral Shoppe", in 2011.

 The album helped popularize the vaporwave genre throughout the Internet.

 Since then, she has continued to release music through Bandcamp and other online platforms, and has recently collaborated with rapper Siddiq on two albums,
"Midnight Run" and "Palacio Del Rio" (both from 2016).
</item 1>

<item 2>
I Remember (AlunaGeorge album)
I Remember is the second studio album by English electronic music duo AlunaGeorge.

 It was released on 16 September 2016 by Island Records.

 The album includes collaborations with Popcaan, Zhu, Leikeli47, Dreezy, Flume and Pell.

 The album's lead single, "I'm in Control", was released in February 2016, featuring vocals from rapper Popcaan, and peaked at number 39 on the UK Singles
Chart.
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
Zhu (musician)
Steven Zhu (born 1989), known professionally as Zhu ( or ), is a Chinese American electronic musician and singer who has been active since the beginning of
2014, signed to Mind of a Genius Records and Columbia Records.

 Until mid-2014, Zhu remained anonymous, asking to be judged by his music alone.

 His debut album, "Generationwhy", was released on 29 July 2016.
</item 4>

<item 5>
King Electric
King Electric is Chris Isepp and Peter Hartwig.

 They took their first musical attempts at the end of 1980s / beginning 1990s and first met each other at the upcoming and flourishing Viennese techno scene
back in '92.

 While Hartwig was touring with the well known noise-pop band the Basket Boys, Isepp made first attentions as Electro DJ & Producer.
</item 5>

<item 6>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 6>

<item 7>
Candy Chen
Candy Chen (陳斯亞 born January 3, 1993) is a dancer, actress, host, singer, rapper and a model.

 She is featured on the Taiwanese variety shows: Blackie's Teenage Club and Blackie Lollipop.

 Candy was a member of the Taiwanese girl group Hey Girl from 2010 to 2011, and a member of from 2013 to 2016.

 She is 1/8th Dutch.

 She changed her name to Chen Yi-Ling in July 2012 to follow her mother's maiden name.

 Her English name becomes Nina Chen.

 In November 2013, she changed her name to Candy Chen (English Name) and Chen Si Ya (Chinese Name).
</item 7>

<item 8>
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
</item 8>

<item 9>
Graph query row:
person: Steven Zhu
birth_date: 1989
relationship: None
album: I Remember
</item 9>

<item 10>
Cherushii
Cherushii (Chelsea Faith Dolan) (September 14, 1983 – December 2, 2016) was an American electronic music producer, DJ, live performer, and radio host based in
San Francisco.

 Influenced by house music, Detroit techno, and disco, she released several recordings on the independent dance music label 100% Silk, including "Queen of
Cups", "Far Away So Close", and "Memory of Water", reissued in 2017.

 KQED named Cherushii's album "Meow Wolf’s Arcade Soundtracks: Wiggy’s Plasma Plex" one of the 10 best bay area albums of 2016.

 While Cherushii has received little recognition from mainstream media, her music is of enormous influence in the underground music scene.

 Fellow electronic musician Maria Minerva has called Cherushii criminally underrated.
</item 10>
=== END ITEM id=73fb398e5226f5db ===

=== ITEM id=67c085bc3b05f0a8 ===
Gold sentences and facts:
Sentence 0: Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca O'Brien and Lauren Dark.
  Fact 0.0: Mediastan is a documentary film.
  Fact 0.1: Mediastan was released in 2013.
  Fact 0.2: Mediastan is about the 2010 United States diplomatic cables leak.
  Fact 0.3: Mediastan was directed by Johannes Wahlström.
  Fact 0.4: Mediastan was produced by Julian Assange.
  Fact 0.5: Mediastan was produced by Rebecca O'Brien.
  Fact 0.6: Mediastan was produced by Lauren Dark.
Sentence 1: Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he founded in 2006.
  Fact 1.0: Julian Paul Assange was born Julian Paul Hawkins.
  Fact 1.1: Julian Paul Assange was born on 3 July 1971.
  Fact 1.2: Julian Paul Assange is Australian.
  Fact 1.3: Julian Paul Assange is a computer programmer.
  Fact 1.4: Julian Paul Assange is the founder of WikiLeaks.
  Fact 1.5: WikiLeaks is an organisation.
  Fact 1.6: Julian Paul Assange founded WikiLeaks in 2006.

Retrieved items:
<item 1>
Aryanto Yuniawan
Aryanto Yuniawan (born June 18, 1977 in Magelang; age 40) is Chief Executive Officer of PT Mataram Surya Visi Sinema, a multimedia and animation movie company
located in Jogja Indonesia.

 Ary (his nickname) is best known as a director of Battle of Surabaya, the first 2D animation movie in Indonesia which got some awards i.e. Gold Remi Award USA
2016, GrandPrize Winner of SICAF 2016 in South Korea, Winner of Nice International Filmmaker Festival 2017 in France, Winner of Nioda International Film
Festival in India and others.

 He is also acclaimed as a writer, a producer, and currently working on several articles and books related to his job and hobby.

 He started his career from directing and producing independent animation films including music videos.

 He was trusted to be one of the juries of several film festivals such as Festival Film Indonesia (FFI 2016), Noida International Film Festival (NIFF 2016), and
Seoul International Cartoon and Animation Festival (SICAF 2017).
</item 1>

<item 2>
Ebba Witt-Brattström
Ebba Witt-Brattström (born 1953) is a Swedish scholar in comparative literature.

 She is Professor of Literature and head of department at Södertörn University outside Stockholm, and a well-known feminist.

 In the 1970s she was a member of the feminist organisation Grupp 8, and in 2005 she was one of the founders of the feminist political organisation and party
Feministiskt Initiativ, although she later distanced herself from the organisation and criticized its alleged strong left-wing tendencies.
</item 2>

<item 3>
Loveleen Tandan
Loveleen Tandan is an Indian film director and casting director.

 She is the "Co-Director: India" of the four time Golden Globe, seven time BAFTA Award and eight Academy Award winning (including best picture) "Slumdog
Millionaire" (2008), for which she shared a New York Film Critics Online Award, Rotterdam International Film Festival Award and Amanda Awards, Norway, of "Best
Director" with Danny Boyle.

 She has also been the Casting Director for several other films, including the Golden Lion winning and Golden Globe nominated "Monsoon Wedding" (2001) and the
BAFTA Award nominated "Brick Lane" (2007).

 She has been a Casting Consultant for the Gotham Award and Independent Spirit Award nominated film "The Namesake" (2007).
</item 3>

<item 4>
Shamal Sabri
Shamal Sabri (Kurdish: شه‌مال سه‌بری), (born 9 November 1985), is an independent Kurdish award-winning film producer and Duhok International Film Festival
Artistic Director, graduated from Duhok University with BA in English Literature.

.

 He was born in the city of Duhok (Kurdish: دهوك) in Iraqi Kurdistan.

 Apart from being one of the co-founders of Duhok IFF in 2009, he has worked in many other fields of video productions, produced many commercials and music
videos in NV Production and Vin TV.

 Throughout his film carrier, he has contributed in many award-winning films as an assistant director, production manager, assistant producer and line producer.

 His latest contributions were in many films such as "Before Snowfall" by Hisham Zaman, "The Swallow" by Mano Khalil, "Memories on Stone" by Shawkat Amin Korki
and Michael by Kurdo Duski.
</item 4>

<item 5>
Faraz Anwar (film)
Faraz Anwar – Documentary is a 2005 documentary film directed and produced by Haroon Sheikh.

 The documentary film follows the experience of Faraz Anwar, lead guitarist and vocalist of the Pakistani progressive rock band Mizraab, in the music industry.

 The documentary was released on November, 2005, by Poison Wood Productions.
</item 5>

<item 6>
Mediastan
Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca
O'Brien and Lauren Dark.
</item 6>

<item 7>
Shajoon Kariyal
He debuted as a director with "Rajaputhran" (1996), starring Suresh Gopi, Shobhana and Vikram.

 He has directed many films, including the commercially successful "Thachiledathu Chundan" (1999) and the critically acclaimed "Vadakkumnadhan" (2006).

 After "Vadakkumnadhan", he planned two films, "Raman Police" and "Talkies", but both the projects did not work out.

 In 2012, he directed "Chettayees" which he also co-produced, as one of the five partners of the newly launched production house Thakkaali Films.

 His latest film is "Sir C. P." (2015).
</item 7>

<item 8>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 8>

<item 9>
Julian Assange
Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he
founded in 2006.

 He has won accolades including the Sam Adams Award and Martha Gellhorn Prize for Journalism.
</item 9>

<item 10>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 10>
=== END ITEM id=67c085bc3b05f0a8 ===

=== ITEM id=91c0c7e5eec43863 ===
Gold sentences and facts:
Sentence 0: George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
  Fact 0.0: George Albert Smith Sr. was born on April 4, 1870.
  Fact 0.1: George Albert Smith Sr. died on April 4, 1951.
  Fact 0.2: George Albert Smith Sr. was an American religious leader.
  Fact 0.3: George Albert Smith Sr. served as the eighth president of The Church of Jesus Christ of Latter-day Saints (LDS Church).
Sentence 1: Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.
  Fact 1.0: Wong Jing was born on 3 May 1955.
  Fact 1.1: Wong Jing is a Hong Kong film director.
  Fact 1.2: Wong Jing is a Hong Kong film producer.
  Fact 1.3: Wong Jing is a Hong Kong actor.
  Fact 1.4: Wong Jing is a Hong Kong presenter.
  Fact 1.5: Wong Jing is a Hong Kong screenwriter.

Retrieved items:
<item 1>
What of the Mormons?
Hinckley wrote "What of the Mormons?"

 while he was employed as the Executive Secretary to the Church Radio, Publicity, and Mission Literature Committee.

 As part of the celebration of the 100th anniversary of the Mormon pioneers arriving in the Salt Lake Valley, Hinckley had been asked by church president George
Albert Smith to write a book that would introduce the LDS Church to non-members.
</item 1>

<item 2>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 2>

<item 3>
D. Arthur Haycock
David Arthur Haycock (September 4, 1916 – February 25, 1994) was a personal secretary to several twentieth-century presidents of The Church of Jesus Christ of
Latter-day Saints (LDS Church), including George Albert Smith, Joseph Fielding Smith, Harold B. Lee, Spencer W. Kimball and Ezra Taft Benson.
</item 3>

<item 4>
Albert Smith Medal
The Albert Smith Memorial Medal is the award given to the Man of the Match in the final of the Camanachd Cup, the blue riband trophy of the sport of shinty.

 It has been presented every year since 1972 by the Smith family of Fort William in honour of Albert Smith Sr. (Born in 1888, Lochuanagan, Fort Augustus.)

 The widow of his only son, Albert Smith Jr. currently presents the medal.

 Albert Sr.'s grandson, Victor Smith is a former player for Fort William Shinty Club but never won the medal despite featuring as a key player in several Fort
William wins.
</item 4>

<item 5>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 5>

<item 6>
The X-Rays
The X-Rays (also known as The X-Ray Fiend) is a 1897 British short silent comedy film, directed by George Albert Smith, featuring a courting couple exposed to
X-rays.

 The trick film, according to Michael Brooke of BFI Screenonline, "contains one of the first British examples of special effects created by means of jump cuts"
Smith employs the jump-cut twice; first to transform his courting couple via "X rays," dramatized by means of the actors donning black bodysuits decorated with
skeletons, and then to return them to normal.

 The couple in question were played by Smith's wife Laura Bayley and Tom Green (a Brighton comedian).
</item 6>

<item 7>
George A. Smith
George Albert Smith (June 26, 1817 – September 1, 1875) (known throughout his life as George A. Smith) was an early leader in the Latter-day Saint movement.

 He served in the Quorum of the Twelve Apostles and as a member of the First Presidency of The Church of Jesus Christ of Latter-day Saints (LDS Church).
</item 7>

<item 8>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 8>

<item 9>
Ba-Jia-Jiang
Ba Jia Jiang (八家將) originated from the Chinese folk beliefs and myths, and usually generally refers to eight members of the godly realm.

 The general understanding of the origin of Ba Jia Jiang is that it is derived from the existence of eight generals who performed exorcism of evil spirits for
the Wufu Emperor (五福大帝).

 These eight generals became revered as the gods of the underworld, and are represented as the bodyguards or attendants for the temples of the nether Gods such
as the Dongyue Emperor (東獄大帝), Yama (King of Hell, 閻羅王) and Cheng Huang (the City Gods, 城隍).

 Gradually Ba Jia Jiang evolved to appear also as bodyguards to Wang Ye (Royal Lord, 王爺) and Matsu (媽祖), and at many other temples.

 Later on, believers at those temples dressed up as Ba Jia Jiang in order to defend the Gods.
</item 9>

<item 10>
Laura Bayley
Laura Eugenia Bayley (1864 – 1938) was a British actress and filmmaker, active in the Brighton School of early cinema pioneers.

 Born in Ramsgate, Bayley performed onstage in Victorian burlesques, revues, and pantomimes, often with her three sisters.

 After marrying the showman George Albert Smith, she entered the world of early experiments with motion picture film; she played main roles in many of the most
important films Smith made between 1897 and 1903, including "The Kiss in the Tunnel" (1899) and "Mary Jane's Mishap" (1903).
</item 10>
=== END ITEM id=91c0c7e5eec43863 ===

=== ITEM id=89e905cbac2333dc ===
Gold sentences and facts:
Sentence 0: Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
  Fact 0.0: Miltoniopsis is a genus of orchids.
  Fact 0.1: Miltoniopsis is abbreviated Mltnps in horticultural trade.
  Fact 0.2: Miltoniopsis is native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
Sentence 1: Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few species extending north into temperate regions of Asia.
  Fact 1.0: Strobilanthes is a genus of about 350 species of flowering plants.
  Fact 1.1: Strobilanthes is in the family Acanthaceae.
  Fact 1.2: Strobilanthes is mostly native to tropical Asia and Madagascar.
  Fact 1.3: A few Strobilanthes species extend north into temperate regions of Asia.

Retrieved items:
<item 1>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 1>

<item 2>
Hemigraphis
Hemigraphis is a genus of plants in the family Acanthaceae, consisting of about 30 species native to tropical Asia.

 "Hemigraphis" is sometimes included in the genus "Strobilanthes"
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 4>

<item 5>
Miltoniopsis vexillaria
Miltoniopsis vexillaria ("the flag-like "Miltoniopsis"") is a species of epiphytic orchid in the genus Miltoniopsis.
</item 5>

<item 6>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 6>

<item 7>
Nepenthes
Nepenthes ( ), also known as tropical pitcher plants, is a genus of carnivorous plants in the monotypic family Nepenthaceae.

 The genus comprises roughly 150 species, and numerous natural and many cultivated hybrids.

 They are mostly liana-forming plants of the Old World tropics, ranging from South China, Indonesia, Malaysia and the Philippines; westward to Madagascar (two
species) and the Seychelles (one); southward to Australia (three) and New Caledonia (one); and northward to India (one) and Sri Lanka (one).

 The greatest diversity occurs on Borneo, Sumatra, and the Philippines, with many endemic species.

 Many are plants of hot, humid, lowland areas, but the majority are tropical montane plants, receiving warm days but cool to cold, humid nights year round.

 A few are considered tropical alpine, with cool days and nights near freezing.

 The name "monkey cups" refers to the fact that monkeys have been observed drinking rainwater from these plants.
</item 7>

<item 8>
Miltoniopsis
Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.

 This genus comprises 5 species.

 Miltoniopsis's common name is Pansy Orchid.
</item 8>

<item 9>
Strobilanthes japonica
Strobilanthes japonica is a flowering herbaceous perennial plant from Asia, one of around 350 plants of the genus "Strobilanthes".

 The 20–50 cm ornamental plant is cultivated in Japan and China, and blooms in autumn with 1.5 cm purple to white funnel-shaped flowers.
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
=== END ITEM id=89e905cbac2333dc ===

=== ITEM id=7a51de69bb8f4de2 ===
Gold sentences and facts:
Sentence 0: The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).
  Fact 0.0: The Washington City Canal operated from 1815 until the mid-1850s.
  Fact 0.1: The Washington City Canal operated in Washington, D.C.
  Fact 0.2: The Washington City Canal connected the Anacostia River to Tiber Creek.
  Fact 0.3: The Washington City Canal connected the Anacostia River to the Potomac River.
  Fact 0.4: The Washington City Canal later connected the Anacostia River to the Chesapeake and Ohio Canal (C&O).
  Fact 0.5: The Anacostia River was called the 'Eastern Branch' at that time.
Sentence 1: The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River to Hagerstown, Indiana.
  Fact 1.0: The Whitewater Canal was built between 1836 and 1847.
  Fact 1.1: The Whitewater Canal spanned a distance of seventy-six miles.
  Fact 1.2: The Whitewater Canal stretched from Lawrenceburg, Indiana.
  Fact 1.3: Lawrenceburg, Indiana is on the Ohio River.
  Fact 1.4: The Whitewater Canal stretched to Hagerstown, Indiana.

Retrieved items:
<item 1>
Washington City Canal
The Washington City Canal operated from 1815 until the mid-1850s in Washington, D.C. The canal connected the Anacostia River, called the "Eastern Branch" at
that time, to Tiber Creek, the Potomac River, and later the Chesapeake and Ohio Canal (C&O).

 The canal fell into disuse in the late 19th century and the city government covered over or filled in various sections.
</item 1>

<item 2>
The entries cover historic U.S. canals, waterways, water systems, and sites. Transport canals include Beaver–Erie (136 miles), Delaware (60), Whitewater (76),
and Susquehanna–Tidewater (43), plus interstate and regional routes; many declined with railroad competition. The Dismal Swamp Canal, opened in 1805 and closed
in 2016, served the Intracoastal Waterway; nearby park and refuge protect the swamp.
</item 2>

<item 3>
Cincinnati and Whitewater Canal Tunnel
The Cincinnati and Whitewater Canal Tunnel is a historic yet abandoned canal tunnel in the southwestern corner of the U.S. state of Ohio.

 Located within the village of Cleves near Cincinnati, it was constructed in 1837 as part of the Whitewater Canal system.

 Since the canal's closure, it has largely been forgotten, but it has been designated a historic site.
</item 3>

<item 4>
Duck Creek Aqueduct
Duck Creek Aqueduct, also known as the Metamora Aqueduct and Whitewater Canal Aqueduct, is a historic aqueduct carrying the Whitewater Canal over Duck Creek in
Metamora Township, Franklin County, Indiana.

 It is the only surviving covered wood aqueduct in the United States as such it is a form of covered bridge.

 It is a single-span Burr through truss aqueduct and measures approximately 90 feet long, 25 feet wide, and 25 feet deep.

 The original aqueduct was built between 1839 and 1843.

 The present structure was built in 1846, after the original aqueduct was washed out in a flood.

 It was strengthened in 1868, and repaired in 1901.

 After abandonment and deterioration, the Duck Creek Aqueduct was restored to its present appearance in 1946-1949.
</item 4>

<item 5>
Dismal Swamp Canal
The Dismal Swamp Canal is located along the eastern edge of the Great Dismal Swamp in Virginia and North Carolina in the United States.

 It is the oldest continually operating man-made canal in the United States, opened in 1805, and closed in October 2016.

 It is part of the Intracoastal Waterway, an inland route, which parallels the east coast and offers boaters shelter from the Atlantic Ocean from Manasquan
Inlet, New Jersey, to Brownsville, Texas.

 The route runs through bays, lakes, rivers, streams, and canals, and includes the Intracoastal Waterway running from Norfolk, Virginia, to the Florida Keys.
</item 5>

<item 6>
Pennsylvania and Ohio Canal
The Pennsylvania and Ohio Canal, also known as the P & O Canal, the Cross Cut Canal and the Mahoning Canal was a shipping canal which operated from 1840 until
1877 (though the canal was completely abandoned by 1872).

 It was unique in that it served to connect canals in two states (the Ohio and Erie Canal in Ohio and the Beaver and Erie Canal in Pennsylvania) and was funded
by private interests.
</item 6>

<item 7>
New York’s Erie Canal heritage includes Lock 52 (1849–53), canal buildings and archaeological sites; the 1836–42 Broad Street Aqueduct later became a road and
subway route. The 1905 song “Low Bridge” recalls mule-powered canal traffic. The 525-mile state canal system includes the Erie, Oswego, Cayuga–Seneca and
Champlain canals. The Genesee River and tributaries Black Creek and Oatka Creek drain western New
</item 7>

<item 8>
Whitewater Canal
The Whitewater Canal, which was built between 1836 and 1847, spanned a distance of seventy-six miles and stretched from Lawrenceburg, Indiana on the Ohio River
to Hagerstown, Indiana.
</item 8>

<item 9>
Brunswick–Altamaha canal
The Brunswick–Altamaha Canal was a 12-mile-long canal built to connect the Altamaha River to the city of Brunswick, Georgia at the Turtle River to transport
goods from Brunswick's port to areas farther north and inland.

 It included locks on both ends of the canal.

 The canal was originally proposed in 1798 and after two failed charters (1826 and 1830), construction was started in 1836 by Thomas Butler King.

 The project was estimated to cost $450,000.

 The canal was finished and opened in 1854; however, by that time the canal was obsolete because of new railroads.

 The canal was closed by 1860.

 The Columbian mammoth was discovered during the construction of the canal.
</item 9>

<item 10>
Susquehanna and Tidewater Canal
The Susquehanna and Tidewater Canal between Wrightsville, Pennsylvania, and Havre de Grace, Maryland, at the head of Chesapeake Bay, provided an interstate
shipping alternative to 19th-century arks, rafts, and boats plying the difficult waters of the lower Susquehanna River.

 Built between 1836 and 1840, it ran 43 mi along the west bank of the river and rendered obsolete an older, shorter canal along the east bank.

 Of its total length, 30 mi were in Pennsylvania and 13 mi in Maryland.

 Though rivalry between Philadelphia, Pennsylvania, and Baltimore, Maryland, delayed its construction, the finished canal brought increased shipments of coal
and other raw materials to both cities from Pennsylvania's interior.

 Competition from railroads was a large factor in the canal's decline after 1855.

 Canal remnants, including a lock keeper's house, have been preserved in Maryland, and locks 12 and 15 have been preserved in Pennsylvania.
</item 10>
=== END ITEM id=7a51de69bb8f4de2 ===

=== ITEM id=c899bb76cf4a4a4e ===
Gold sentences and facts:
Sentence 0: Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.
  Fact 0.0: Vytautas Kevin Gerulaitis was born on July 26, 1954.
  Fact 0.1: Vytautas Kevin Gerulaitis died on September 17, 1994.
  Fact 0.2: Vytautas Kevin Gerulaitis was an American professional tennis player.
Sentence 1: Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.
  Fact 1.0: Heather Miriam Watson was born on 19 May 1992.
  Fact 1.1: Heather Miriam Watson is a British tennis player.
  Fact 1.2: Heather Miriam Watson is a Wimbledon Mixed Doubles Champion.

Retrieved items:
<item 1>
1977 Italian Open (tennis)
The 1977 Italian Open was a combined men's and women's tennis tournament that was played by men on outdoor clay courts at the Foro Italico in Rome, Italy.

 The men's tournament was part of the Colgate-Palmolive Grand Prix circuit while the women's tournament was part of the Colgate Series.

 The tournament was held from 16 May through 22 May 1977.

 The singles titles were won by eight-seeded Vitas Gerulaitis and fifth-seeded Janet Newberry who earned $21,000 and $6,000 first-prize money respectively.

 Gerulaitis competed despite being contracted to play for the Indiana Loves World Team Tennis (WTT) franchise and was fined $19,000 for failing to play Björn
Borg in the weekend of the final.
</item 1>

<item 2>
Vitas Gerulaitis
Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.

 In 1975, Gerulaitis won the men's doubles title at Wimbledon, partnering with Sandy Mayer.

 He won the men's singles title at one of the two Australian Open tournaments held in 1977 (Gerulaitis won the tournament that was held in December, while
Roscoe Tanner won the earlier January tournament).

 Gerulaitis also won two Italian Open titles, in 1977 and 1979, and the WCT Finals in Dallas, in 1978.
</item 2>

<item 3>
Heather Watson
Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.

 She is a former British No.1 and current British No. 2 behind Johanna Konta.
</item 3>

<item 4>
Michael Venus (tennis)
Michael Venus (born 16 October 1987) is a New Zealand professional tennis player.

 He reached a career high singles ranking of 274 in July 2011, but is much better known as a leading doubles player with a huge serve.

 His highest ranking in this discipline is 12, gained in July 2017.

 He won the final of the 2017 French Open partnering with American Ryan Harrison, and followed that with a runner-up finish in the 2017 US Open mixed doubles
with Chan Hao-ching of Chinese Taipei as his partner.
</item 4>

<item 5>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 5>

<item 6>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 6>

<item 7>
Sharon Walraven
Sharon Walraven (born 19 June 1970, Schaesberg) is a Dutch wheelchair tennis player.

 She became paraplegic at age 23 after complications following a fall while she was ice-skating.

 She has won seven Grand Slams doubles titles partnering compatriot Esther Vergeer.

 At the 2008 Paralympics in Beijing she won the Gold medal in the women's doubles competition.

 At the 2000 Paralympics in Sydney she won a Silver medal in the women's singles competition.

 Walraven has a highest ranking of No.2 in singles and No.1 in doubles.
</item 7>

<item 8>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 8>

<item 9>
1974 Stadthalle Open – Singles
Vitas Gerulaitis won in the final 6–4, 3–6, 6–3, 6–2 against Andrew Pattison.
</item 9>

<item 10>
1979 World Championship Tennis Finals – Singles
Vitas Gerulaitis was the defending champion but lost in the semifinals to Björn Borg.
</item 10>
=== END ITEM id=c899bb76cf4a4a4e ===

=== ITEM id=7c867e6820ad36fe ===
Gold sentences and facts:
Sentence 0: Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit single in 1986 with "Love Missile F1-11" – and six other EMI single releases.
  Fact 0.0: Martin Degville was born 27 January 1961 in Walsall.
  Fact 0.1: Martin Degville is the lead singer of the UK pop band Sigue Sigue Sputnik.
  Fact 0.2: Martin Degville is a co-songwriter of Sigue Sigue Sputnik.
  Fact 0.3: Sigue Sigue Sputnik had a worldwide hit single in 1986 with 'Love Missile F1-11'.
  Fact 0.4: Sigue Sigue Sputnik has six other EMI single releases.
Sentence 1: Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.
  Fact 1.0: Paul Richard 'Richie' Furay was born May 9, 1944.
  Fact 1.1: Richie Furay is an American singer.
  Fact 1.2: Richie Furay is an American songwriter.
  Fact 1.3: Richie Furay is a Rock & Roll Hall of Fame member.
  Fact 1.4: Richie Furay is best known for forming Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin.
  Fact 1.5: Richie Furay is best known for forming Poco with Jim Messina, Rusty Young, George Grantham, and Randy Meisner.

Retrieved items:
<item 1>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 1>

<item 2>
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 2>

<item 3>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 3>

<item 4>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 4>

<item 5>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 5>

<item 6>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
</item 6>

<item 7>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 7>

<item 8>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 8>

<item 9>
Poco: The Songs of Richie Furay
Poco: The Songs of Richie Furay is a compilation album consisting of songs by Richie Furay during his tenure with the band Poco, released in 1980.
</item 9>

<item 10>
Lonesome Pine Fiddlers
The Lonesome Pine Fiddlers (1938- 1966) were an early bluegrass band which included such notable "first generation" bluegrass musicians as Ezra Cline, Bobby
Osborne, Paul Williams, Melvin Goins, Charlie Cline, Curly Ray Cline, Larry Richardson and for a short time Jimmy Martin.

 The group was started by Ezra Cline and Curly Ray Cline and was originally named "Cousin Ezra and the Lonesome Pine Fiddlers".

 The Clines came from a large family consisting of musically talented people.

 Ray and Charlie's father, Charlie, was a talented banjo player and the women in the family, Geraldine and Bobbi, were great singers.

 For reasons unknown, Bobbi and Geraldine never joined the band on the road but often joined in at home, especially when notable Country singers, such as Bill
Monroe, Lester Flatt and Earl Scruggs, and Hank Williams, came visiting.

 None of them ever had a music lesson yet excelled on every instrument they touched.
</item 10>
=== END ITEM id=7c867e6820ad36fe ===
