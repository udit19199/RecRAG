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

=== ITEM id=d35692204fbe0f6e ===
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
Christopher Guest
Christopher Haden-Guest, 5th Baron Haden-Guest (born February 5, 1948), simply known as Christopher Guest, is an English-American screenwriter, composer,
musician, director, actor, and comedian who holds dual British and American citizenship.

 Guest is most widely known in Hollywood for having written, directed and starred in his series of comedy films shot in mock-documentary (mockumentary) style.

 Many scenes and character backgrounds in Guest's films are written and directed, although actors have no rehearsal time and the ensemble improvise scenes while
filming them.

 The series of films began with "This Is Spinal Tap" (directed by Rob Reiner), and continued with "Waiting for Guffman", "Best In Show", "A Mighty Wind", "For
Your Consideration", and "Mascots".
</item 1>

<item 2>
The 'Burbs
The 'Burbs is a 1989 American comedy thriller film directed by Joe Dante starring Tom Hanks, Bruce Dern, Carrie Fisher, Rick Ducommun, Corey Feldman, Wendy
Schaal and Henry Gibson.

 The film was written by Dana Olsen, who also has a cameo in the movie.

 The film pokes fun at suburban environments and their eccentric dwellers.
</item 2>

<item 3>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 3>

<item 4>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 4>

<item 5>
Thom Eberhardt
Thom Eberhardt (born March 7, 1947) is an American film director, producer and screenwriter.

 Eberhardt has won two awards and two nominations.

 He is most noted for his work on "Captain Ron", "Honey, I Blew Up the Kid", and the cult classic "Night of the Comet".
</item 5>

<item 6>
Pizza 73
Pizza 73 is a Canadian restaurant chain that offers a number of different styles of pizza, along with chicken wings.

 It has been operated by Pizza Pizza since 2007.

 Toronto-based Pizza Pizza had acquired the restaurant for a total of $CAN70.2 million.

 There are 89 locations throughout Western Canada, which include the provinces of British Columbia, Alberta, and Saskatchewan.

 The restaurant's name originates from its original phone number: 473–7373.

 Founded by David Tougas and Guy Goodwin in 1985, Pizza 73 is headquartered in Edmonton, Alberta, Canada.
</item 6>

<item 7>
Jebb Fink
Jebb Fink is an American-Canadian stand-up comedian and television personality.

 Best known as a reporter and host for "the Big Breakfast" on A-Channel, Calgary.

 He won a Gemini as co-creator of the CBC comedy "An American in Canada" and was a weather presenter on "Your City" for City TV Calgary, he also hosted the
short-lived late night talk show "Global Late Night" for the Global Television Network in 2004.
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
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 9>

<item 10>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 10>
=== END ITEM id=d35692204fbe0f6e ===

=== ITEM id=b29874542cf568b4 ===
Gold sentences and facts:
Sentence 0:  The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
  Fact 0.0: The women's events include the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park.
  Fact 0.1: The women's events include the Roxy Pro at Sunset Beach.
  Fact 0.2: The women's events include the Billabong Pro Maui at Honolua Bay, Maui.
Sentence 1: Haleʻ iwa is a North Shore community and census-designated place (CDP) in the Waialua District of the island of Oʻ ahu, City and County of Honolulu.
  Fact 1.0: Haleʻiwa is a North Shore community.
  Fact 1.1: Haleʻiwa is a census-designated place (CDP).
  Fact 1.2: Haleʻiwa is in the Waialua District of the island of Oʻahu.
  Fact 1.3: Haleʻiwa is in the City and County of Honolulu.

Retrieved items:
<item 1>
Reef Hawaiian Pro
The Reef Hawaiian Pro is a surfing event held annually on the North Shore, Oahu, Hawaii.

 It is currently the first leg of the Triple Crown of Surfing.

 Surfers compete for a share of $135,000 (men) or $35,000 (women) prize money.

 Since it carries a 6-star rating, the event is critical for those who want to qualify for the ASP World Tour.
</item 1>

<item 2>
Dana Brown
Dana Brown (born December 11, 1959 in Dana Point, California) is an American surfer and filmmaker, and is the oldest son of filmmaker Bruce Brown.

 His films include "The Endless Summer Revisited" (2000) which is made up of unused footage from "The Endless Summer" (1964) and "The Endless Summer II" (1994),
as well as some original interviews with the stars of those films.

 His first all-original film was "Step Into Liquid" (2003) followed by a documentary on the Baja 1000 titled "Dust to Glory" (2005).

 In 2009, he debuted a new film called "Highwater" during the 100th anniversary of the Santa Monica Pier; the film follows life on the North Shore and the
surfers who compete in the Vans Triple Crown of Surfing.

 In 2014, the movie "On Any Sunday, The Next Chapter" continues the saga of motocross documentaries which began with the 1972 Academy Award for Documentary
Feature nominated film "On Any Sunday" (1971).
</item 2>

<item 3>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 3>

<item 4>
2012 Monterrey Open
The 2012 Monterrey Open was a women's tennis tournament played on outdoor hard courts.

 It was the 4th edition of the Monterrey Open and was an International tournament on the 2012 WTA Tour.

 It took place at the Sierra Madre Tennis Club in Monterrey, Mexico, from 20 to 26 February.

 Tímea Babos won the singles title.
</item 4>

<item 5>
Santa Barbara International Marathon
The Santa Barbara International Marathon and Half Marathon is an annual race that takes place in Goleta, California and Santa Barbara, California.

 The inaugural race was on December 6, 2009.

 For the last five years the Santa Barbara City College has been the official host for the finish line and general operations.

 Rusty and June Snow are the co-directors of the race with Dan Campbell as the technical director.

 The Snow's moved from the East Coast and wanted to bring a marathon to Santa Barbara.

 After a few years of getting the race approved and organized it was finally set to happen in 2009, Rusty Snow states, “I learned to be stubborn from running
marathons.”

 The Santa Barbara International Marathon course serves as a Boston Marathon qualifier.
</item 5>

<item 6>
Ocean City Open
The Ocean City Open is a defunct tennis tournament that was played on the Grand Prix tennis circuit in 1977.

 The event was held in Ocean City, Maryland.

 Vitas Gerulaitis won the singles title while Alex Metreveli and Bill Scanlon partnered to win the doubles title.
</item 6>

<item 7>
Triple Crown of Surfing
The Vans Triple Crown of Surfing is a Hawaiian specialty series of professional surfing events, offering three events to men and three events to women.

 For the men, those events are the Reef Hawaiian Pro at Haleiwa Ali'i Beach Park; the O'Neill World Cup of Surfing at Sunset Beach; and the Billabong Pipeline
Masters at the Banzai Pipeline.

 The women's events are the Vans Hawaiian Pro at Haleiwa Ali'i Beach Park; the Roxy Pro at Sunset Beach; and the Billabong Pro Maui at Honolua Bay, Maui.
</item 7>

<item 8>
The material surveys international Games, athletes, teams and sports history. Guadalajara’s 2011 Pan American Games hosted 6,000 athletes from 42 nations;
Toronto’s 2015 Games involved 18 communities. At the 2015 Pacific Games, Papua New Guinea led women’s powerlifting, Nauru men’s, Tuvalu won its first gold, and
Nicolas N’Godrela claimed three tennis golds. Other highlights include Iona’s MAAC success, NHL
</item 8>

<item 9>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 9>

<item 10>
World Match Racing Tour
The World Match Racing Tour is an annual match racing tour and World Championship in the match racing event of sailing.

 The World Match Racing Tour has "special event"-status from the International Sailing Federation.

 The Tour awards over US$1.5 million in prize money, with points awarded at each event culminating in the crowning of the ISAF Match Racing World Champion.

 A US$500,000 overall prize pool for the Championship is now on offer.
</item 10>
=== END ITEM id=b29874542cf568b4 ===

=== ITEM id=93e9a11f9eafaadf ===
Gold sentences and facts:
Sentence 0: Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist, pianist, and primary songwriter for the Christian rock band Relient K.
  Fact 0.0: Matthew Arnold 'Matt' Thiessen was born August 12, 1980.
  Fact 0.1: Matt Thiessen is a Canadian-American musician, singer, and songwriter.
  Fact 0.2: Matt Thiessen is known for being co-founder of the Christian rock band Relient K.
  Fact 0.3: Matt Thiessen is known for being lead singer of the Christian rock band Relient K.
  Fact 0.4: Matt Thiessen is known for being guitarist of the Christian rock band Relient K.
  Fact 0.5: Matt Thiessen is known for being pianist of the Christian rock band Relient K.
  Fact 0.6: Matt Thiessen is known for being primary songwriter for the Christian rock band Relient K.
Sentence 1: George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.
  Fact 1.0: George Thorogood was born February 24, 1950.
  Fact 1.1: George Thorogood is an American musician, singer, and songwriter.
  Fact 1.2: George Thorogood is from Wilmington, Delaware.

Retrieved items:
<item 1>
My Girlfriend (Relient K song)
"My Girlfriend" is a song by the Christian rock band Relient K, released on their self-titled first album.

 The song originally appeared as "Marilyn Manson Ate My Girlfriend" on the band's demo album, "All Work and No Play".

 The song is about Marilyn Manson eating Matt Thiessen's girlfriend.

 Thiessen wrote this song when he was 15 years old.

 Thiessen has said that he wrote it because of a female friend, who lived eight hours away in Pennsylvania, who he would talk to about many things including
spiritual matters such as where God was taking them in the future.

 His friend would later turn from Christian music to Nine Inch Nails and Marilyn Manson.

 In an interview with CCM Magazine Thiessen stated "through this she changed her lifestyle [and] what she believed in."

 His friend would later be expelled from school and would be kicked out of her house and sent to a youth detention center.

 Thiessen would later state "She felt that Christianity was stupid and just this big hypocrisy.

 Being young and impressionable, I just wrote this little, stupid song, but that was the way I dealt with it—writing this song about how she got so consumed by
Marilyn Manson."
</item 1>

<item 2>
George Thorogood and the Destroyers (album)
George Thorogood and the Destroyers is the self-titled debut album by American blues rock band George Thorogood and the Destroyers, released in 1977.

 Consisting mostly of covers of blues hits, it included a medley of John Lee Hooker's "House Rent Boogie" and "One Bourbon, One Scotch, One Beer", the latter a
song written by Rudy Toombs for Amos Milburn, and later covered by Hooker.
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
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 4>

<item 5>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 5>

<item 6>
Better Than the Rest
Better Than the Rest is the third album (mini-album) of songs by George Thorogood and the Destroyers, recorded in 1974 and released in 1979.

 At the time the songs were recorded, Thorogood was an unknown artist who was not signed to a record label.

 In 1979, after Thorogood had released his first two official albums on Rounder Records, "Better Than The Rest" was released by MCA.

 In 1986, the songs from this album were released on compact disc with the title "Nadine" using a different track order.
</item 6>

<item 7>
Rebuild (song)
"Rebuild" is a song written by Jon Foreman of Switchfoot and Matt Thiessen of Relient K.

 The song was written in conjunction with the bands' 2007 Appetite for Construction Tour, featuring members from all three bands on tour, Switchfoot, Relient K,
and Ruth.

 Thiessen sang second vocals, and Dustin Ruth of Ruth played harmonica.
</item 7>

<item 8>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 8>

<item 9>
More George Thorogood and The Destroyers
More George Thorogood and the Destroyers is the fourth album by George Thorogood and The Destroyers, released in 1980.

 An alternate name for the album is I'm Wanted.
</item 9>

<item 10>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 10>
=== END ITEM id=93e9a11f9eafaadf ===

=== ITEM id=16dfcaf6cd35cad0 ===
Gold sentences and facts:
Sentence 0: Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.
  Fact 0.0: Margaret Landon was born September 7, 1903.
  Fact 0.1: Margaret Landon died December 4, 1993.
  Fact 0.2: Margaret Landon was an American writer.
  Fact 0.3: Margaret Landon is best remembered for 'Anna and the King of Siam'.
  Fact 0.4: 'Anna and the King of Siam' is Margaret Landon's best-selling novel, published in 1944.
  Fact 0.5: 'Anna and the King of Siam' is a novel about the life of Anna Leonowens.
  Fact 0.6: 'Anna and the King of Siam' eventually sold over a million copies.
  Fact 0.7: 'Anna and the King of Siam' was translated into more than twenty languages.
Sentence 1:  In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.
  Fact 1.0: In 1950, Margaret Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II.
  Fact 1.1: Richard Rodgers and Oscar Hammerstein II created the musical 'The King and I' from Margaret Landon's book.
Sentence 2:  A later work, "Never Dies the Dream", appeared in 1949.
  Fact 2.0: 'Never Dies the Dream' is a later work.
  Fact 2.1: 'Never Dies the Dream' appeared in 1949.
Sentence 3: Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.
  Fact 3.0: Dame Elizabeth Rosemond Taylor, DBE, was born February 27, 1932.
  Fact 3.1: Dame Elizabeth Rosemond Taylor, DBE, died March 23, 2011.
  Fact 3.2: Dame Elizabeth Rosemond Taylor was a British-American actress.
  Fact 3.3: Dame Elizabeth Rosemond Taylor was a businesswoman.
  Fact 3.4: Dame Elizabeth Rosemond Taylor was a humanitarian.
Sentence 4:  She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.
  Fact 4.0: Elizabeth Taylor began her career as a child actress in the early 1940s.
  Fact 4.1: Elizabeth Taylor was one of the most popular stars of classical Hollywood cinema in the 1950s.
Sentence 5:  She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.
  Fact 5.0: Elizabeth Taylor continued her career successfully into the 1960s.
  Fact 5.1: Elizabeth Taylor remained a well known public figure for the rest of her life.
Sentence 6:  In 1999, the American Film Institute named her the seventh-greatest female screen legend.
  Fact 6.0: In 1999, the American Film Institute named Elizabeth Taylor the seventh-greatest female screen legend.

Retrieved items:
<item 1>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 1>

<item 2>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 2>

<item 3>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 3>

<item 4>
Laura Bayley
Laura Eugenia Bayley (1864 – 1938) was a British actress and filmmaker, active in the Brighton School of early cinema pioneers.

 Born in Ramsgate, Bayley performed onstage in Victorian burlesques, revues, and pantomimes, often with her three sisters.

 After marrying the showman George Albert Smith, she entered the world of early experiments with motion picture film; she played main roles in many of the most
important films Smith made between 1897 and 1903, including "The Kiss in the Tunnel" (1899) and "Mary Jane's Mishap" (1903).
</item 4>

<item 5>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 5>

<item 6>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 6>

<item 7>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 7>

<item 8>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 8>

<item 9>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 9>

<item 10>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 10>
=== END ITEM id=16dfcaf6cd35cad0 ===

=== ITEM id=33d548e7612d4091 ===
Gold sentences and facts:
Sentence 0: Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.
  Fact 0.0: Eucryphia is a small genus.
  Fact 0.1: Eucryphia consists of trees and large shrubs.
  Fact 0.2: Eucryphia is native to the south temperate regions of South America.
  Fact 0.3: Eucryphia is native to coastal eastern Australia.
Sentence 1:  Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.
  Fact 1.0: Eucryphia is sometimes placed in a family of its own, the Eucryphiaceae.
  Fact 1.1: More recent classifications place Eucryphia in the Cunoniaceae.
Sentence 2:  There are seven species, two in South America and five in Australia, and several named hybrids.
  Fact 2.0: Eucryphia has seven species.
  Fact 2.1: Two Eucryphia species are in South America.
  Fact 2.2: Five Eucryphia species are in Australia.
  Fact 2.3: Eucryphia has several named hybrids.
Sentence 3: Lepidozamia is a genus of two species of cycad, both endemic to Australia.
  Fact 3.0: Lepidozamia is a genus of cycad.
  Fact 3.1: Lepidozamia has two species.
  Fact 3.2: Both Lepidozamia species are endemic to Australia.

Retrieved items:
<item 1>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 1>

<item 2>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 2>

<item 3>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 3>

<item 4>
Eriogonum
Eriogonum is the scientific name for a genus of flowering plants in the family Polygonaceae.

 The genus is found in North America and is known as wild buckwheat.

 This is a highly species-rich genus, and indications are that active speciation is continuing.

 It includes some common wildflowers such as the California buckwheat ("Eriogonum fasciculatum").
</item 4>

<item 5>
Lophodermium
Lophodermium is a genus of fungi within the family Rhytismataceae.

 The genus contains 145 species and has a global distribution.

 Species of this genus are usually observed producing zone lines, conidiomata and ascomata on dead fallen leaves, but at least some are known to colonize living
leaves.

 In many cases they then live inside the colonized leaf as a symptomless endobiont, where they are regarded as detritivores utilising dead plant matter.

 In a few cases they may kill all or part of the leaf prematurely, and there is a substantial literature dealing with those species as plant pathogens.

 The genus infects many different plant families but with a notable concentration in the family Pinaceae; many "Lophodermium" species are restricted to a single
host genus (or even species), but some, particularly those infecting grasses, may infect several genera.

 Some are economically important plant pathogens, such as those that cause needlecast disease in European Black Pine, Scots Pine and Red Pine in forestry and
christmas tree plantations.
</item 5>

<item 6>
Eryngium
Eryngium is a genus of flowering plants in the family Apiaceae.

 There are about 250 species.

 The genus has a cosmopolitan distribution, with the center of diversity in South America.

 Common names include eryngo and amethyst sea holly (though the genus is not related to the true hollies, "Ilex").
</item 6>

<item 7>
Leitneria
Leitneria floridana (corkwood), the sole species in the genus Leitneria, is a deciduous dioecious shrub or small tree, found only in the southeastern United
States states of Arkansas, Florida, Georgia, Missouri and Texas.
</item 7>

<item 8>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 8>

<item 9>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 9>

<item 10>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 10>
=== END ITEM id=33d548e7612d4091 ===

=== ITEM id=27dc321c90e2d412 ===
Gold sentences and facts:
Sentence 0: The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
  Fact 0.0: The Albanian Coalition 'Perspective' is an Albanian political party.
  Fact 0.1: The Albanian Coalition 'Perspective' is in Montenegro.
Sentence 1: Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.
  Fact 1.0: Albanians in Montenegro constitute 4.91% of the county's total population.
  Fact 1.1: The Albanian name for Albanians in Montenegro is 'Shqiptarët e Malit të Zi'.

Retrieved items:
<item 1>
Darza
Darza (Albanian: "Darzë" , Montenegrin and ) is a village in the Ulcinj Municipality, southeastern Montenegro.

 It is a multi-ethnic settlement, inhabited by Montenegrins, Serbs and Albanians.

 According to the 2003 census, the total population was 119.
</item 1>

<item 2>
Albania
Albania is a unitary parliamentary constitutional republic with the capital in Tirana, the country's largest city and main economic and commercial centre,
followed by Durrës.

 The country's other major cities include Vlorë, Sarandë, Shkodër, Berat, Korçë, Gjirokastër and Fier.
</item 2>

<item 3>
Albania
Albania ( , ; Albanian: "Shqipëri/Shqipëria" ; Gheg Albanian: "Shqipni/Shqipnia or Shqypni/Shqypnia" ), officially the Republic of Albania (Albanian: "Republika
e Shqipërisë" , ] ), is a country in Southern and Southeastern Europe.

 The country spans 28,748 km2 and had a total population of almost 3 million people as of 2016 .

 Albania is located in the southwestern part of the Balkan Peninsula, bordered by Montenegro to the northwest, Kosovo to the northeast, the Republic of
Macedonia to the east, and Greece to the south and southeast.

 The country has a coastline on the northern shore of the Mediterranean Sea, the Adriatic Sea to the west and the Ionian Sea to the southwest, forming the
Albanian Riviera.

 Albania is less than 72 km from Italy, across the Strait of Otranto which connects the Adriatic Sea to the Ionian Sea.
</item 3>

<item 4>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 4>

<item 5>
Minority languages of Montenegro
European Charter for Regional or Minority Languages entered into force in Montenegro in June 2006, following the independence of Montenegro from the State Union
of Serbia and Montenegro on 3 June 2006.

 Constitution of Montenegro from 2007 states that Montenegrin language is official language of country, while Bosnian, Croatian, Serbian and Albanian are
languages in official use.

 The Constitution states that languages in official use are those of groups that form at least 1% of the population of Montenegro, as per 2003 population
census.

 The Law on National Minorities specifies that the percentage of members of national minorities in total population of the local government should be 15% in
order for their language and script to be introduced in official use.

 Media whose founder is Montenegro government are obliged to broadcast news, cultural, educational, sports and entertainment programs in minority languages.

 Minorities and their members have the right to education in their language in regular and vocational education.
</item 5>

<item 6>
Albanians
Albanians (Albanian: "Shqiptarët" ) are a nation and ethnic group, native to Albania, Kosovo and other countries who share a common culture, ancestry and speak
the Albanian language as a native tongue.

 Legally, the term is used to refer to the citizens of the Republic of Albania.

 Ethnic Albanians speak the Albanian language and more than half of ethnic Albanians live in Albania and Kosovo.

 A large Albanian population lives in Greece, Italy, the Republic of Macedonia, with smaller Albanian populations located in Serbia and Montenegro.
</item 6>

<item 7>
Albanian Coalition &quot;Perspective&quot;
The Albanian Coalition "Perspective" is an Albanian political party in Montenegro.
</item 7>

<item 8>
Albanians in Montenegro
Albanians in Montenegro (, Albanian: "Shqiptarët e Malit të Zi" ) constitute 4.91% of the county's total population.

 Albanians of Montenegro are Ghegs who mainly live in southeastern and eastern Montenegro, mainly in the following municipalities: Ulcinj (71% of population),
Plav (19%), Bar (6%), Podgorica (5%) and Rožaje (5%).
</item 8>

<item 9>
Albania is a parliamentary republic in the Balkans, bordering Montenegro, Kosovo, North Macedonia and Greece, with Adriatic and Ionian coastlines; Tirana is its
capital. Albanians, including Gheg speakers, live across the region, notably in Kosovo, North Macedonia and Montenegro. Montenegro recognizes several minority
languages and supports minority-language media and education. Serbia gained Kosovo and Vardar Macedonia in the Balkan Wars. The text also briefly describes
Turks, Azerbaijanis
</item 9>

<item 10>
Albanians in the Republic of Macedonia
Albanians are the largest ethnic minority in the Republic of Macedonia.

 Of the 2,022,547 citizens of Macedonia, 509,083, or 25.2%, are Albanian according to the latest national census in 2002.

 The Albanian minority lives mostly in the north-western part of the country.

 The largest Albanian communities are in the municipalities of Tetovo (70.3% of the total population), Gostivar (66.7%), Debar (58.1%), Struga (56.8%), Kičevo
(54.5%), Kumanovo (25.8%) and Skopje (20.4%).
</item 10>
=== END ITEM id=27dc321c90e2d412 ===

=== ITEM id=cd5753ee1519aaf9 ===
Gold sentences and facts:
Sentence 0: Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.
  Fact 0.0: Kim Dong-jun's Hangul name is 김동준.
  Fact 0.1: Kim Dong-jun's Hanja name is 金桐俊.
  Fact 0.2: Kim Dong-jun was born February 11, 1992.
  Fact 0.3: Kim Dong-jun is a South Korean singer.
  Fact 0.4: Kim Dong-jun is a South Korean actor.
Sentence 1: Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.
  Fact 1.0: Charlie Harper was born David Charles Perez.
  Fact 1.1: Charlie Harper was born on 25 May 1944.
  Fact 1.2: Charlie Harper was born in Hackney, London.
  Fact 1.3: Charlie Harper is a British singer.
  Fact 1.4: Charlie Harper is a British songwriter.

Retrieved items:
<item 1>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 1>

<item 2>
Graph query row:
person: Kim Dong-jun
occupation: ['singer', 'actor']
nationality: South Korean
birth_date: February 11, 1992
death_date: None
gender: None
relationship: MEMBER_OF
related_entity: South Korean boyband
</item 2>

<item 3>
Kim Dong-jun (footballer)
Kim Dong-jun (Korean: 김동준 , born 19 December 1994) is a South Korean footballer who plays as a goalkeeper for Yonsei University and South Korea national team.
</item 3>

<item 4>
Graph query row:
person: Charlie Harper
occupation: ['singer', 'songwriter']
nationality: British
birth_date: 25 May 1944
death_date: None
gender: None
relationship: BORN_IN
related_entity: Hackney, London
</item 4>

<item 5>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 5>

<item 6>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 6>

<item 7>
Charlie Harper (singer)
Charlie Harper (born David Charles Perez, 25 May 1944, Hackney, London) is a British singer and songwriter.

 He is best known as the lead singer from the punk band U.K. Subs.
</item 7>

<item 8>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 8>

<item 9>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 9>

<item 10>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 10>
=== END ITEM id=cd5753ee1519aaf9 ===

=== ITEM id=f67e94766dd8709f ===
Gold sentences and facts:
Sentence 0: Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.
  Fact 0.0: Jack and Jill is an American magazine for children.
  Fact 0.1: Jack and Jill is published bimonthly.
  Fact 0.2: Jack and Jill is for children 7 to 12 years old.
  Fact 0.3: The title of Jack and Jill comes from the nursery rhyme of the same name.
Sentence 1: Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.
  Fact 1.0: Paper, stylized as PAPER, is a New York City-based magazine.
  Fact 1.1: Paper is an independent magazine.
  Fact 1.2: Paper focuses on fashion.
  Fact 1.3: Paper focuses on popular culture.
  Fact 1.4: Paper focuses on nightlife.
  Fact 1.5: Paper focuses on music.
  Fact 1.6: Paper focuses on art.
  Fact 1.7: Paper focuses on film.

Retrieved items:
<item 1>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
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
Jack and Jill (magazine)
Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.

 It features stories and educational activities.
</item 3>

<item 4>
Mighty Jill Off
Mighty Jill Off is a 2D platform independently developed freeware video game designed by Anna Anthropy, with art by James Harvey and music by Andrew Toups.

 It stars a submissive named Jill, who has a boot fetish and is forced to climb up a tower after her Queen kicks her down it as punishment.

 Jill does this by jumping and slowly descending over obstacles.

 Jill can be defeated in one hit by these obstacles, but will return to the last check point.

 The game serves as an homage to the 1986 arcade game "Mighty Bomb Jack".

 It had follow-ups, such as "Mighty Jill Off - Jill Off Harder Edition" and "Jill Off With One Hand".

 Jill made a cameo appearance in the 2010 video game "Super Meat Boy" as a playable character.
</item 4>

<item 5>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 5>

<item 6>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 6>

<item 7>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 7>

<item 8>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 8>

<item 9>
The entries survey magazines, broadcasters, websites, and media demographics. Publications range from children’s nature titles and women’s magazines to
political, religious, literary, and LGBTQ+ outlets; profiles include Rosie O’Donnell. Radio and TV examples span US local stations, regional Spectrum News
channels, and international youth/children’s networks. Adult-video sites include xHamster, YouPorn, and XVideos. Advertisers’ “key demographic” often means
adults 18–49, though targets vary
</item 9>

<item 10>
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 10>
=== END ITEM id=f67e94766dd8709f ===

=== ITEM id=add382be9a28c44b ===
Gold sentences and facts:
Sentence 0:  Editors are based in Silver Spring, Maryland and Seoul, Korea.
  Fact 0.0: Editors are based in Silver Spring, Maryland.
  Fact 0.1: Editors are based in Seoul, Korea.
Sentence 1:  The magazine is headquartered in Baltimore, Maryland.
  Fact 1.0: The magazine is headquartered in Baltimore, Maryland.

Retrieved items:
<item 1>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 1>

<item 2>
Seventh-day Adventist education
The Seventh-day Adventist educational system is part of the Seventh-day Adventist Church and overseen by the General Conference of Seventh-day Adventist located
in Silver Spring, Maryland.The educational system is the second-largest Christian school-system in the world, after the Roman Catholic system.
</item 2>

<item 3>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 3>

<item 4>
Adventist HealthCare
Adventist HealthCare is a not-for-profit health services organization based in Gaithersburg, Maryland that employs more than 6,200 people and provides
healthcare for more than 400,000 individuals in the community each year.

 The primary service area for Adventist HealthCare is the Washington, D.C. metropolitan area.

 Despite similar names, it is not a part of the California-based Adventist Health, or the Florida-based Adventist Health System.
</item 4>

<item 5>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 5>

<item 6>
Ellendale, Delaware
Ellendale is a town in Sussex County, Delaware, United States.

 The population was 381 at the 2010 census, an increase of 16.5% since 2000.

 It is part of the Salisbury, Maryland-Delaware Metropolitan Statistical Area.

 Ellendale is the "Gateway to Delaware's Resort Beaches" because it is the town located on U.S. Highway 113, the resort area's westernmost border, and Delaware
Route 16, the resort area's northernmost border with the eastern border being the Delaware Bay and Atlantic Ocean and the southern border being the state line
with Maryland.

 Ellendale is home to the Philadelphia Bible College and the Harbor Christian Academy.
</item 6>

<item 7>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 7>

<item 8>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 8>

<item 9>
Adventist World
Adventist World is a monthly international magazine of the Seventh-day Adventist Church published by the Review and Herald Publishing Association.

 Editors are based in Silver Spring, Maryland and Seoul, Korea.
</item 9>

<item 10>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 10>
=== END ITEM id=add382be9a28c44b ===

=== ITEM id=2f374f7bdf618410 ===
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
Lakes Region (New Hampshire)
The Lakes Region of New Hampshire is the mid-state region surrounding Lake Winnipesaukee, Winnisquam Lake, Squam Lake, and Newfound Lake.

 The area comprises all of Belknap County, the southern portion of Carroll County, the eastern portion of Grafton County, and the northern portions of Strafford
County and Merrimack County.

 The largest municipality is the city of Laconia.
</item 1>

<item 2>
KXMB-TV
KXMB-TV, channel 12, is the CBS-affiliated television station for Bismarck, North Dakota.

 It broadcasts a high definition digital signal on VHF channel 12 (or virtual channel 12.1 via PSIP) from a transmitter near St. Anthony.

 The station can also be seen on Midcontinent cable channel 11 in Bismarck-Mandan, and cable channel 12 in most other areas.

 There is a high definition feed provided on Midcontinent digital channel 611.
</item 2>

<item 3>
KKBN
KKBN (93.5 FM) The Cabin a radio station since 1986 broadcasting a Country music format since March 2000.

 Licensed to Twain Harte, California, United States, the station serves the greater Mother Lode area comprising Tuolumne and Calaveras Counties with a combined
population of approximately 110,000.

 The station is currently owned by Clarke Broadcasting Corporation.
</item 3>

<item 4>
KWXL-LP
KWXL-LP (98.7 FM) is a high school radio station broadcasting a variety format.

 Licensed to Tucson, Arizona, United States, the station serves the Tucson area.

 The station is currently owned by the Tucson Unified School District.

 KWXL-LP is Tucson's only high school radio station.

 It is also a news radio station for students who attend Pueblo High Magnet School.

 Students broadcasting over the radio station are enrolled in a "Writing/Reporting for Broadcasting" class with instructor Sarah Walson.

 Originally started by Douglas Potter, who retired in 2006.

 Listeners can also find FM 98.7, KWXL on iTunes.
</item 4>

<item 5>
Crystal Lake (Gilmanton, New Hampshire)
Crystal Lake is a 455 acre water body located in Belknap County in the Lakes Region of central New Hampshire, United States, in the town of Gilmanton.

 Crystal Lake is at the head of the Suncook River watershed.

 Water flows into Crystal Lake from the Belknap Range through Manning and Sunset lakes to the north.

 Water from Manning Lake enters Crystal Lake via Nelson Brook.

 Other waterways flowing into Crystal Lake include Wasson Brook and Mill Brook.

 Water flows out of a dam from the southern end of Crystal Lake via the Suncook Lakes and Suncook River to the Merrimack River.

 Shoreline development along Crystal Lake consists primarily of summer cottages, with a few year-round residents.

 Belknap Mountain and Mount Major can be seen from the shoreline.
</item 5>

<item 6>
WXMK
WXMK (105.9 FM) is a radio station broadcasting a Hot Adult Contemporary format.

 Licensed to Dock Junction, Georgia, USA, the station serves the Brunswick area.

 Better known as Magic 105.9 "The Golden Isles Number 1 Hit Music Station."

 The station is currently owned by Golden Isles Broadcasting, LLC.
</item 6>

<item 7>
KRBT
KRBT (1340 AM) is a radio station broadcasting a sports format.

 Licensed to Eveleth, Minnesota, United States, the station serves the Iron Range area.

 The station is currently owned by Range Broadcasting, Inc., and features programming from the Red Zone Radio Network.

 Originally a stand alone music station with call letters WEVE, then a simulcast music station with WEVE-FM, the station changed it's call letters to KRBT for
"Range's Best Talk" in June of 1998 and discontinued simulcast with WEVE-FM to become a talk station.

 The format change to sports/talk was made following the death of previous owner, Lew Latto, Iron Range Broadcasting, Inc.

 President on August, 24 2011 and subsequent sale to Red Rock Radio in March 2012, and has been retained by the new owner as of July 2017, Range Broadcasting,
Inc.
</item 7>

<item 8>
KVKI-FM
KVKI-FM (96.5 FM) is a radio station broadcasting an adult contemporary music format.

 Licensed to Shreveport, Louisiana, United States, the station serves the Shreveport area.

 The station is currently owned by Townsquare Media.

 Its studios are shared with its other five sister stations in West Shreveport (one mile west of Shreveport Regional Airport), and the transmitter is in
Blanchard, Louisiana.

 The call letters were originally KBCL, and later became religious station KEPT, before the call letters were changed to KVKI to be similar to the call letters
of the popular KVIL radio station in Dallas, Texas.
</item 8>

<item 9>
WLMR
WLMR (1450 AM) is a radio station broadcasting a religious format.

 Licensed to Chattanooga, Tennessee, USA, the station serves the Chattanooga area.

 The station is currently owned by Wilkins Communications Network, Inc. and features programming from USA Radio Network.

 In the early 1980s, the station was automated and played country music from studios in a strip mall on Brainerd Rd.

 It was also Chattanooga's first talk radio station when it held the WZRA call letters.

 WZRA was the first home of Jeff Styles, and also featured well known personalities such as Kelly McCoy and Robert T. Nash.
</item 9>

<item 10>
WVNH
WVNH (91.1 FM) is a radio station broadcasting a Christian radio format.

 Licensed to Concord, New Hampshire, United States, the station serves the Concord area.

 The station is currently owned by New Hampshire Gospel Radio, Inc.

 Programming is simulcast on WANH (91.5 FM) in Meredith, serving the Lakes Region.
</item 10>
=== END ITEM id=2f374f7bdf618410 ===

=== ITEM id=ae98a588836821dc ===
Gold sentences and facts:
Sentence 0:  Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
  Fact 0.0: The unnamed person in this passage has other voice-over roles
  Fact 0.1: The unnamed person voiced Dr. Drakken in Kim Possible
  Fact 0.2: The unnamed person voiced Motor Ed in Kim Possible
  Fact 0.3: The unnamed person voiced Brother Blood in Teen Titans
  Fact 0.4: The unnamed person voiced Aquaman in an unnamed work
  Fact 0.5: The unnamed person voiced Rico in The Penguins of Madagascar
  Fact 0.6: The unnamed person voiced Ogre in an unnamed work
  Fact 0.7: The unnamed person voiced Niblet in Pound Puppies
  Fact 0.8: The unnamed person voiced the Scotsman in Samurai Jack
  Fact 0.9: The unnamed person voiced Shnitzel in Chowder
Sentence 1: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 1.0: Chowder is an American animated television series
  Fact 1.1: Chowder was created by C. H. Greenblatt
  Fact 1.2: Chowder was created for Cartoon Network

Retrieved items:
<item 1>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 1>

<item 2>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 2>

<item 3>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 3>

<item 4>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 4>

<item 5>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 5>

<item 6>
List of Invader Zim characters
"Invader Zim" is an American animated television series created by Jhonen Vasquez and originally aired on Nickelodeon.

 The recurring cast includes voice actors Richard Steven Horvitz, Rodger Bumpass, Kevin McDonald ("The Kids in the Hall") and John de Lancie ("").

 Vasquez voices parts in the show.

 The show was cancelled early in its run and some episodes were unfinished.

 The show ran for two seasons before its cancellation.
</item 6>

<item 7>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 7>

<item 8>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 8>

<item 9>
Harvey Beaks
Harvey Beaks is an American animated television series created by C.H. Greenblatt for Nickelodeon.

 The series premiered on March 28, 2015 following the 2015 Kids' Choice Awards.
</item 9>

<item 10>
Adventure Time (pilot)
"Adventure Time" is an animated short created by Pendleton Ward, as well as the pilot to the Cartoon Network series of the same name.

 The short follows the adventures of Pen (voiced by Zack Shada), a human boy, and his best friend Jake (voiced by John DiMaggio), a dog with magical powers to
change shape and grow and shrink at will.

 In this episode, Pen and Jake have to rescue Princess Bubblegum (voiced by Paige Moss) from the antagonistic Ice King (voiced by John Kassir).
</item 10>
=== END ITEM id=ae98a588836821dc ===

=== ITEM id=930af15347e46d5f ===
Gold sentences and facts:
Sentence 0: Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.
  Fact 0.0: Cars 2 is a film.
  Fact 0.1: Cars 2 was released in 2011.
  Fact 0.2: Cars 2 is American.
  Fact 0.3: Cars 2 is computer-animated.
  Fact 0.4: Cars 2 is an action comedy spy film.
  Fact 0.5: Cars 2 was produced by Pixar Animation Studios.
  Fact 0.6: Cars 2 was produced for Walt Disney Pictures.
Sentence 1: Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway musicals "Hamilton" and "In the Heights".
  Fact 1.0: Lin-Manuel Miranda was born January 16, 1980.
  Fact 1.1: Lin-Manuel Miranda is American.
  Fact 1.2: Lin-Manuel Miranda is a composer, lyricist, playwright, and actor.
  Fact 1.3: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical Hamilton.
  Fact 1.4: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical In the Heights.
Sentence 2:  He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".
  Fact 2.0: Lin-Manuel Miranda co-wrote the songs for Disney's Moana soundtrack.
  Fact 2.1: The Moana soundtrack is from 2016.
  Fact 2.2: Lin-Manuel Miranda is set to star in Disney's upcoming film Mary Poppins Returns.

Retrieved items:
<item 1>
Cars 2
Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the sequel to 2006's "Cars", and features the voices of Owen Wilson, Larry the Cable Guy, Michael Caine, Emily Mortimer, John Turturro, and Eddie Izzard.

 In the film, race car Lightning McQueen and tow truck Mater head to Japan and Europe to compete in the World Grand Prix, but Mater becomes sidetracked with
international espionage.

 The film was directed by John Lasseter, written by Ben Queen, and produced by Denise Ream.
</item 1>

<item 2>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 2>

<item 3>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 3>

<item 4>
Cabinet Battle
"Cabinet Battle #1", "Cabinet Battle #2", and "Cabinet Battle #3" are songs written for Act II of the musical "Hamilton", based on the life of Alexander
Hamilton, which premiered on Broadway in 2015.

 Lin-Manuel Miranda wrote both the music and lyrics to the song.
</item 4>

<item 5>
Cars 2 (video game)
Cars 2 (also known as Cars 2: The Video Game) is a racing game developed by Avalanche Software.

 It is based on the film "Cars 2", the sequel to the 2006 film "Cars".

 Originally announced at E3 2011, the game was released by Disney Interactive on major platforms in North America on June 21, 2011, and in Australia 2 days
later.

 The game was released in Europe on July 22, 2011.

 The PlayStation 3 version of the game was reported to feature stereoscopic 3D gameplay.

 A Nintendo 3DS version was released on November 1, 2011 (the same day the film was released on DVD and Blu-ray).

 A PlayStation Portable version was released in North America on November 8, 2011.

 It was also released in Europe on November 4, 2011.

 It was the last Pixar game to be released for the PlayStation Portable.
</item 5>

<item 6>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 6>

<item 7>
Your Obedient Servant (song)
"Your Obedient Servant" is the twentieth song from Act 2 of the musical "Hamilton", based on the life of Alexander Hamilton, which premiered on Broadway in
2015.

 Lin-Manuel Miranda wrote both the music and lyrics to the song.

 The song is sung by the characters of Hamilton and Aaron Burr, originally performed by Miranda and Leslie Odom Jr., respectively.
</item 7>

<item 8>
Moana (soundtrack)
Moana: Original Motion Picture Soundtrack is the soundtrack to the 2016 Disney animated film "Moana".

 The soundtrack was released by Walt Disney Records on November 19, 2016.

 It features songs written by Lin-Manuel Miranda, Mark Mancina and Opetaia Foa'i, with lyrics in English, Samoan, Tokelauan, and Tuvalu.

 The two-disc deluxe edition includes the score, which was composed by Mancina, as well as demos, outtakes and instrumental karaoke tracks.
</item 8>

<item 9>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 9>

<item 10>
Lin-Manuel Miranda, creator and star of *Hamilton* and *In the Heights*, wrote its music and lyrics and co-wrote songs for Disney’s *Moana*. His honors include
a Pulitzer, three Tonys, two Grammys, an Emmy, and a MacArthur Fellowship. The *Hamilton* cast album broke sales and chart records; its songs include “Your
Obedient Servant” and the Cabinet Battles. The text also covers Disney and Muppet soundtracks, *The M
</item 10>
=== END ITEM id=930af15347e46d5f ===
