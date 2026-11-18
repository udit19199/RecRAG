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

=== ITEM id=f365aa1f571f60ac ===
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
Ronn Owens
Ronn Owens (born Ronald Lowenstein on October 17, 1945, in New York City), is an American talk radio host.

 Owens was the sole remaining weekday talk radio host on KGO in San Francisco after KGO was reorganized as an all-news station in December 2011.
</item 1>

<item 2>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 2>

<item 3>
Mike Hogan (sportscaster)
Mike Hogan (born 1963 in Kingston, Ontario) is a Canadian sportscaster who is a sports talk radio host on TSN Radio 1050 in Toronto.

 He also serves as the play-by-play voice for the Toronto Argonauts of the Canadian Football League for that station's gameday broadcasts.

 Hogan was formerly a long time sports talk radio host on Toronto station CJCL (AM), branded "The Fan 590", until he was relieved of those duties on June 24,
2010.

 On 2017 Oscar Vanity fair party 1st live broadcast Mike Hogan was the podcast hosts.
</item 3>

<item 4>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
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
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 6>

<item 7>
Sean Hannity
Sean Patrick Hannity (born December 30, 1961) is an American talk show host, author, and conservative political commentator.

 Hannity is the host of "The Sean Hannity Show", a nationally syndicated talk radio show.

 He also hosts a cable news show, "Hannity", on Fox News Channel.
</item 7>

<item 8>
David Gold (talk radio host)
David Gold is an American conservative talk radio host.

 Gold, who has been a talk radio host since the mid-1970s, one of the first U.S. conservative talk radio hosts.

 His philosophy has been characterized as in the conservative/libertarian vein.
</item 8>

<item 9>
Joe Scarborough
Charles Joseph Scarborough ( ; born April 9, 1963) is an American cable news and talk radio host.

 He is currently the co-host of "Morning Joe" on MSNBC, and previously hosted "Scarborough Country" on the same channel.

 Scarborough was previously a lawyer and a politician, and served in the United States House of Representatives from 1995 to 2001 as a Republican from the 1st
district of Florida.
</item 9>

<item 10>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 10>
=== END ITEM id=f365aa1f571f60ac ===

=== ITEM id=a512f188e9c8098d ===
Gold sentences and facts:
Sentence 0: Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
  Fact 0.0: Chalk and Cheese was a short-lived sitcom television show
  Fact 0.1: Chalk and Cheese featured Michael Crawford
  Fact 0.2: Chalk and Cheese featured Robin Hawdon
Sentence 1: Michael Patrick Smith, {'1': ", '2': ", '3': ", '4': "} (born 19 January 1942), known as Michael Crawford, is an English actor, comedian and singer.
  Fact 1.0: Michael Patrick Smith is known as Michael Crawford
  Fact 1.1: Michael Patrick Smith was born 19 January 1942
  Fact 1.2: Michael Crawford is English
  Fact 1.3: Michael Crawford is an actor
  Fact 1.4: Michael Crawford is a comedian
  Fact 1.5: Michael Crawford is a singer

Retrieved items:
<item 1>
Chalk and Cheese
Chalk and Cheese was a short lived sitcom television show, featuring Michael Crawford and Robin Hawdon.
</item 1>

<item 2>
Michael Crawford
Michael Patrick Smith, {'1': ", '2': ", '3': ", '4': "} (born 19 January 1942), known as Michael Crawford, is an English actor, comedian and singer.

 He has received great critical acclaim and won numerous awards during his career, which covers radio, television, film, and stagework on both London's West End
and on Broadway in New York City.
</item 2>

<item 3>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 3>

<item 4>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 4>

<item 5>
Dan Clark
Daniel Gregory Clark (born 3 July 1976 in Bromley, London) is an English actor, comedian, writer, director, and singer.

 He is best known for playing Don Danbury on the BBC Three sitcom "How Not to Live Your Life", which he also wrote, co-produced, and sometimes directed.

 He has been a regular on the British comedy scene as both a sketch and stand-up comedian.
</item 5>

<item 6>
Jonathan Routh
Jonathan Routh, born John Reginald Surdeval Routh, (1927–2008) co-starred in the British version of the television show "Candid Camera" (1960–67) and co-starred
with Germaine Greer and Kenny Everett in a later attempt at a revival, "Nice Time" (1968).

 He published a number of humorous books, and also painted for many years.
</item 6>

<item 7>
A Cream Cracker under the Settee
"A Cream Cracker Under The Settee" is a dramatic monologue written by Alan Bennett in 1987 for television, as part of his "Talking Heads" series for the BBC.

 The series became very popular, moving onto BBC Radio, international theatre, becoming one of the best-selling audio book releases of all time and included as
part of both the A-level and GCSE English syllabus.

 It was the sixth and final episode of the first series of "Talking Heads".
</item 7>

<item 8>
Roger Kitter
Roger Kitter (20 October 1949 – 3 January 2015) was an English actor best known for playing Captain Alberto Bertorelli in series 7 of the British sitcom TV
series "'Allo 'Allo!

".

 He had previously appeared weekly with Lulu throughout the 10-week run of her 1973 BBC1 series "It's Lulu".

 With Kaplan Kaye he also recorded a song "Chalk Dust – the Umpire Strikes Back" using the moniker 'The Brat'.

 Released on the Hansa label, it entered the UK Singles Chart on 10 July 1982; it reached a peak of number 19, and remained in the chart for 8 weeks.

 The song was a Top 10 hit in the Netherlands, Belgium and South Africa, and lampooned John McEnroe complaining about line calls in tennis ("The ball's in,
everyone can see that the ball's in!")

.

 He was also an impressionist, who provided the voice of Tommy Cooper in the Lego 'Kipper' advertisement.
</item 8>

<item 9>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 9>

<item 10>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 10>
=== END ITEM id=a512f188e9c8098d ===

=== ITEM id=47a4dd7c6e4a9649 ===
Gold sentences and facts:
Sentence 0: Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.
  Fact 0.0: Innocence is a 2013 American horror drama film.
  Fact 0.1: Innocence was directed by Hilary Brougher.
  Fact 0.2: Hilary Brougher co-wrote Innocence with Tristine Skyler.
Sentence 1:  The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
  Fact 1.0: The movie Innocence stars Sophie Curtis.
  Fact 1.1: The movie Innocence stars Kelly Reilly.
  Fact 1.2: The movie Innocence stars Graham Phillips.
  Fact 1.3: The movie Innocence stars Linus Roache.
  Fact 1.4: The movie Innocence stars Sarah Sutherland.
  Fact 1.5: The movie Innocence stars Stephanie March.
Sentence 2: Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.
  Fact 2.0: Stephanie Caroline March was born on 23, 1974.
  Fact 2.1: Stephanie Caroline March is an American actress.
Sentence 3:  She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
  Fact 3.0: Stephanie March is best known for her role as Alexandra Cabot.
  Fact 3.1: The role of Alexandra Cabot was in a long-running NBC series.

Retrieved items:
<item 1>
Gloria Garayua
Gloria Garayua (born October 18, 1978 in New York City) is an American film and television actress.

 Garayua made her major film debut in the 2005 comedy film "Fun With Dick and Jane", and is now commonly cast in guest roles on long-running series such as "Six
Feet Under", "Weeds" and "The Shield".

 After being cast in an ongoing role in "Grey's Anatomy", Garayua has played recurring roles in other popular TV shows such as "Cougar Town" and "How to Get
Away with Murder".
</item 1>

<item 2>
Kim Johnston Ulrich
Kim Johnston Ulrich (born March 24, 1955) is an American actress.

 From 1983 to 1986, Ulrich played the role of Diana McColl on "As the World Turns".

 In 1988, she guest-starred in two episodes of "Werewolf".

 In 1990, Ulrich appeared on the TV series "Wings" as Carol, Brian's ex-wife.

 In 1993, Ulrich appeared in the pilot episode of "" as Dr. Antoinette Baines, a villainous scientist.

 She appeared in a third season episode of "Highlander: The Series" in 1995.

 She appeared in the first episode of season 4 of "" as a conniving wife in "Murder by Friendly Fire" in 1996.
</item 2>

<item 3>
Conviction (2006 TV series)
Conviction is an American television drama on NBC that debuted as a mid-season replacement on Friday, March 3, 2006.

 The cast includes Stephanie March reprising her "" role as Alexandra Cabot.

 In the series, Cabot returns to New York City and becomes a Bureau Chief ADA supervising a group of young but talented assistant district attorneys after a
stint in the Witness Protection Program.

 Other cast members include Eric Balfour, Anson Mount, Jordan Bridges, Julianne Nicholson, Milena Govich, and J. August Richards.
</item 3>

<item 4>
In My Sleep
In My Sleep is a 2010 suspense thriller film written, directed and produced by film director Allen Wolf.

 It stars Philip Winchester, Lacey Chabert, Tim Draxl, Abigail Spencer and Kelly Overton.

 It also features Kirsten Vangsness from Criminal Minds in a cameo role and Tony Hale from Veep in a cameo as well.

 The story is about a massage therapist with chronic insomnia who fears he may have murdered a good friend while sleepwalking.
</item 4>

<item 5>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 5>

<item 6>
Allison Miller
Allison Miller (born September 2, 1985) is an American actress.

 She is best known for playing Michelle Benjamin on the NBC series "Kings", Skye Tate on the Fox series "Terra Nova", and Carrie on the NBC series "Go On".

 She starred as Laura Larson on the Syfy television series "Incorporated".
</item 6>

<item 7>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 7>

<item 8>
Stephanie March
Stephanie Caroline March (born (1974--) 23, 1974 ) is an American actress.

 She is best known for her role as Alexandra Cabot in the long-running NBC series, "".
</item 8>

<item 9>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 9>

<item 10>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 10>
=== END ITEM id=47a4dd7c6e4a9649 ===

=== ITEM id=73cfd91f0739acb1 ===
Gold sentences and facts:
Sentence 0: "Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the majority of their members quit and formed Faith No More.
  Fact 0.0: "Quiet in Heaven" / "Song of Liberty" is a double A-side single.
  Fact 0.1: "Quiet in Heaven" / "Song of Liberty" is the only official release by Faith No Man.
  Fact 0.2: Faith No Man is stylised as Faith•No Man•.
  Fact 0.3: The single was released before the majority of Faith No Man's members quit.
  Fact 0.4: After the majority of Faith No Man's members quit, they formed Faith No More.
Sentence 1:  Before settling on their current name in 1982, the band performed under the names Sharp Young Men and later Faith No Man.
  Fact 1.0: The band settled on its current name in 1982.
  Fact 1.1: Before settling on its current name in 1982, the band performed under the name Sharp Young Men.
  Fact 1.2: The band later performed under the name Faith No Man.

Retrieved items:
<item 1>
Jefferson's Tree of Liberty
Jefferson's Tree of Liberty is the tenth album by Jefferson Starship, released on September 2, 2008.

 It is the band's first studio album since 1999's "Windows of Heaven".

 The new album includes cover songs from Irish, American, English, and Latin-American traditions.

 The title is a reference to Thomas Jefferson's quotation, "The tree of liberty must be refreshed from time to time with the blood of patriots and tyrants."

 The idea began in 2003 as "The Cuba Project", which was to include classic protest and folk songs recorded in Cuba.

 In 2008 the album was finally recorded but in California.

 About half of the songs planned for The Cuba Project were used on the final cut, with other songs coming from Jefferson Starship's previous repertoire and
another project band member Paul Kantner had planned called "On the Threshold of Fire."

 The promotional tour for the album began in late June with shows at Larkspur, California followed by tours in the US and Europe before the album's release, and
continued through December 2008 with a further tour in the US and a tour in Japan.
</item 1>

<item 2>
Quiet in Heaven/Song of Liberty
"Quiet in Heaven"/"Song of Liberty" is a double A-side single, and is the only official release, by Faith No Man (stylised as Faith•No Man•), before the
majority of their members quit and formed Faith No More.

 The songs have not appeared in any of the band's compilation albums and have not been performed live since Mike Morris left the band.
</item 2>

<item 3>
Quiet Riot 10
Quiet Riot 10 (also alternatively known as just 10, or Quiet Riot Number 10) is the twelfth studio album by the heavy metal band Quiet Riot, which was released
on June 27, 2014.

 It is their first studio album since 1988's "QR" not to feature longtime and founding vocalist Kevin DuBrow in any newly recorded material, due to his death in
November 2007.

 It is also the band's first album since reuniting in 2010.

 Although a studio album, the final four tracks on "Quiet Riot 10" are live performances taken from some of the band's final shows with DuBrow in 2007.

 Love/Hate vocalist Jizzy Pearl joined the band in November 2013 and performs lead vocals on the six studio tracks.
</item 3>

<item 4>
Faith No More
Faith No More (sometimes abbreviated as FNM) is an American rock band from San Francisco, California, formed in 1979.

 Before settling on their current name in 1982, the band performed under the names Sharp Young Men and later Faith No Man.

 Bassist Billy Gould and drummer Mike Bordin are the longest remaining members of the band, having been involved with Faith No More since its inception.

 The band underwent several lineup changes early in their career, along with some major changes later on.

 The current lineup of Faith No More consists of Gould, Bordin, keyboardist Roddy Bottum, guitarist Jon Hudson and vocalist Mike Patton.
</item 4>

<item 5>
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
</item 5>

<item 6>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 6>

<item 7>
The V.I.P.s (band)
That album was published in 1967 on Liberty Records and contained only five songs from two minutes to more than 15 minutes of psychedelic and almost meditative
state kind of music.
</item 7>

<item 8>
Kiss Me Quiet (song)
"Kiss Me Quiet" is a song recorded by Canadian country singer Jess Moskaluke for her second extended play of the same name (2015).

 It was written by Kelly Archer, Corey Crowder, and Emily Shackelton and was produced by Crowder.

 "Kiss Me Quiet" was released August 28, 2015 as the EP's lead single.

 The song was Moskaluke's first solo top 10 on the Canada Country chart.
</item 8>

<item 9>
Corcovado (song)
"Corcovado" (known in English as "Quiet Nights of Quiet Stars") is a bossa nova song written by Antônio Carlos Jobim in 1960.

 An English lyric was later written by Gene Lees.

 The Portuguese title refers to the Corcovado mountain in Rio de Janeiro.

 Andy Williams recorded the song with English lyrics, reaching #92 in the "Billboard" Hot 100 and #18 in the Hot Adult Contemporary Tracks chart in 1965.

 Also receiving air-play, contemporaneously with Andy Williams' recording of "Quiet Nights," was Kitty Kallen's version.

 Her album, titled "Quiet Nights," was released by 20th Century-Fox Records in 1964.
</item 9>

<item 10>
List of songs recorded by Faith No More
Faith No More, a San Francisco-based band, have recorded approximately 100 songs over the course of their career; this includes material from six studio albums,
one live album, and numerous B-side tracks and out-takes.

 Faith No More were founded in 1981 as Sharp Young Men, and changed their name to Faith No Man before releasing the 1982 double A-side single "Quiet in Heaven"
/ "Song of Liberty".

 Upon assuming the name Faith No More, the band's first two full-length albums, "We Care a Lot" and "Introduce Yourself", were driven mostly by new vocalist
Chuck Mosley and the "metallic guitar" of Jim Martin, blending elements of rap and heavy metal music.

 Mosley was later replaced by Mr. Bungle vocalist Mike Patton, who added lyrics to the already-written music for 1989's "The Real Thing".

 Their next studio album, "Angel Dust", moved away from the band's rap-influenced sound to experiment with different musical genres; a trend which became much
more pronounced on the 1995 album "King for a Day... Fool for a Lifetime".
</item 10>
=== END ITEM id=73cfd91f0739acb1 ===

=== ITEM id=df1030fa3d603581 ===
Gold sentences and facts:
Sentence 0:  It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
  Fact 0.0: The described act consists of a duet.
  Fact 0.1: Dave Hemingway is a vocalist featured in the duet.
  Fact 0.2: Briana Corrigan is a vocalist featured in the duet.
Sentence 1: Briana Corrigan (born 30 May 1965) is a Northern Irish singer.
  Fact 1.0: Briana Corrigan was born on 30 May 1965.
  Fact 1.1: Briana Corrigan is a Northern Irish singer.

Retrieved items:
<item 1>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 1>

<item 2>
Houston (Means I'm One Day Closer to You)
"Houston (Means I'm One Day Closer to You)" is a song written by Larry Gatlin and recorded by American country music group Larry Gatlin & the Gatlin Brothers
Band.

 It was released in September 1983 as the first single from the album ""Greatest Hits Vol.

 II"" then included to first track of ""Not Guilty"" (1984).

 "Houston (Means I'm One Day Closer to You)" was the group's third and last number one on the country chart.

 The single went to number one for two weeks and spent a total of fifteen weeks on the country chart.
</item 2>

<item 3>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 3>

<item 4>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 4>

<item 5>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 5>

<item 6>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 6>

<item 7>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 7>

<item 8>
Graph query row:
person: Briana Corrigan
nationality: Northern Irish
country: None
country_name: None
connected_person: None
person_relationship: None
creative_work: None
work_relationship: None
</item 8>

<item 9>
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
</item 9>

<item 10>
The Closer You Get...
The Closer You Get... is the seventh studio album by country music band Alabama, released in 1983.

 All three singles from this album — "The Closer You Get", "Lady Down on Love" and "Dixieland Delight" — reached Number One on the Hot Country Songs charts in
1983.

 "She Put the Sad in All His Songs" was also recorded by Ronnie Dunn (who formed the duo Brooks & Dunn with Kix Brooks in 1991) and was released by him as a
single in 1983.
</item 10>
=== END ITEM id=df1030fa3d603581 ===

=== ITEM id=583668f473498949 ===
Gold sentences and facts:
Sentence 0:  Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
  Fact 0.0: Akrofuom is located in the Obuasi Municipal district.
  Fact 0.1: The Obuasi Municipal district is in the Ashanti Region of Ghana.
Sentence 1: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.
  Fact 1.0: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

Retrieved items:
<item 1>
Western Region (Ghana)
The Western Region is located in south Ghana, spreads from the Ivory Coast border in the west to the Central region in the east, includes the capital and large
twin city of Sekondi-Takoradi on the coast, coastal Axim, and a hilly inland area including Elubo.

 It includes Ghana's southernmost location, Cape Three Points, where crude oil was discovered in commercial quantities in June 2007.

 The Western Region enjoys a long coastline that stretches from South Ghana's border with Ivory Coast to the Western region's boundary with the Central Region
on the east.
</item 1>

<item 2>
Akrofuom (Ghana parliament constituency)
Akrofuom is one of the constituencies represented in the Parliament of Ghana.

 It elects one Member of Parliament (MP) by the first past the post system of election.

 Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
</item 2>

<item 3>
Tafo
Tafo is a town in Kumasi Metropolitan District in the Ashanti Region of Ghana near the regional capital Kumasi.

 Tafo is the thirtieth most populous settlement in Ghana, in terms of population, with a population of 60,919 people.

 Because of the town's population and housing development in recent years, it is debatable whether Tafo is still regarded as a separate town, or already a
suburb of Kumasi, the capital of the Ashanti region.

 The town is near Kumasi, with a distance of approximately 3.3 kilometers to the center of a similar name sounding village named New Tafo and must be
distinguished from Tafo.

 Tarkwa is located just 4.6 km away from Tafo.

 The city center of Kumasi is located approximately 9.8 kilometers away.

 Tafo is one of the urban constituencies of the Kumasi Metropolitan Assembly, the town's parliamentary candidate shall have one direct seat to the Parliament of
Ghana.
</item 3>

<item 4>
Taifa, Accra
Taifa is a town in the Ga East Municipal District, a district in the Greater Accra Region of south-eastern Ghana near the capital Accra.

 Taifa is the twenty-sixth largest settlement in Ghana, in terms of population, with a population of 68,459 people.

 Taifa is located in the northwest suburbs area of Accra.

 It has a breakpoint on a railway line and a small park located on the northern edge of the location of the Taifa Ghana Atomic Energy Commission.

 At the Ghana 2000 census of 26 March 2000, the population was 26,145 inhabitants living in the city.

 Projections of 1 January 2007 estimated the population to be 48,927 inhabitants.

 In the census of 1984 there was only 1,009 inhabitants.

 The strong population growth of the Town is influenced by, among other things, a large number of illegal immigrants from west African countries who move to
towns and villages near the industrial town of Tema, just to find a job.
</item 4>

<item 5>
Tema
Tema is a city on the Bight of Benin and Atlantic coast of Ghana.

 It is located 25 km east of the capital city; Accra, in the region of Greater Accra, and is the capital of the Tema Metropolitan District.

 As of 2013, Tema is the eleventh most populous settlement in Ghana, with a population of approximately 161,612 people – a marked decrease from its 2005 figure
of 209,000.

 The Greenwich Meridian (00 Longitude) passes directly through the city.

 Tema is locally nicknamed the "Harbour Town" because of its status as Ghana's largest seaport.
</item 5>

<item 6>
Obuasi Municipal District
The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

 The capital of the Obuasi Municipality is Obuasi.
</item 6>

<item 7>
Techimantia
Techimantia is a city located in the Tano District of the Brong Ahafo Region of Ghana.

 It is one of the large cities in the region.

 It is believed to be originally inhabited by Hunters for the Ashanti King from Kumasi.

 These Hunters originally settled in the region after they found that there were lots of elephants in the region which could be hunted for the Ashanti King.The
name of the King is Oti Ampem.Currently,the chief of the town is no more(dead).

The occupation of the people in the town is large scale tomatoes farming.Techimantia is one of the largest towns in the region but there are few productive or
government work in the town.This was attributed to the curse from Okomfo Anokye of the Ashanti Kingdom.
</item 7>

<item 8>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 8>

<item 9>
Ashanti Region
The Ashanti Region is located in south Ghana and is third largest of 10 administrative regions, occupying a total land surface of 24389 km2 or 10.2 per cent of
the total land area of Ghana.

 In terms of population, however, it is the most populated region with a population of 4,780,380 according to the 2010 census, accounting for 19.4% of Ghana’s
total population.

 The Ashanti Region is known for its major gold bar and cocoa production.

 The largest city and regional capital is Kumasi.
</item 9>

<item 10>
Graph query row:
district: Obuasi Municipal district
region: Ashanti Region
</item 10>
=== END ITEM id=583668f473498949 ===

=== ITEM id=1977258f623e422a ===
Gold sentences and facts:
Sentence 0: Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.
  Fact 0.0: Alice Malsenior Walker was born on February 9, 1944.
  Fact 0.1: Alice Malsenior Walker is an American novelist.
  Fact 0.2: Alice Malsenior Walker is an American short story writer.
  Fact 0.3: Alice Malsenior Walker is an American poet.
  Fact 0.4: Alice Malsenior Walker is an American activist.
Sentence 1: Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.
  Fact 1.0: Michael David Herr was born on April 13, 1940.
  Fact 1.1: Michael David Herr died on June 23, 2016.
  Fact 1.2: Michael David Herr was an American writer.
  Fact 1.3: Michael David Herr was an American war correspondent.
  Fact 1.4: Michael David Herr is known as the author of the memoir 'Dispatches', published in 1977.
  Fact 1.5: 'Dispatches' is a memoir of Michael David Herr's time as a correspondent for 'Esquire magazine' from 1967 to 1969.
  Fact 1.6: Michael David Herr was a correspondent during the Vietnam War.

Retrieved items:
<item 1>
Michael Herr
Michael David Herr (April 13, 1940 – June 23, 2016) was an American writer and war correspondent, known as the author of "Dispatches" (1977), a memoir of his
time as a correspondent for "Esquire magazine" (1967–1969) during the Vietnam War.

 The book was called the best "to have been written about the Vietnam War" by "The New York Times Book Review"; novelist John le Carré called it "the best book
I have ever read on men and war in our time."

 Herr later was credited with pioneering the literary genre of the nonfiction novel, along with authors such as Truman Capote, Norman Mailer, and Tom Wolfe.
</item 1>

<item 2>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 2>

<item 3>
Alice Walker
Alice Malsenior Walker (born February 9, 1944) is an American novelist, short story writer, poet, and activist.

 She wrote the critically acclaimed novel "The Color Purple" (1982) for which she won the National Book Award and the Pulitzer Prize for Fiction.

 She also wrote the novels "Meridian" (1976) and "The Third Life of Grange Copeland" (1970), among other works.
</item 3>

<item 4>
Tobe Levin
Alice Walker expressed appreciation for the text that shows worldwide solidarity with the novelist's literary abolition efforts in the early nineties.

 Levin has also teamed up with Maria Kiminta and photographer Britta Radike to publish a memoir and sourcebook, "Kiminta.

 A Maasai's Fight against Female Genital Mutilation."
</item 4>

<item 5>
Dick Herrick
The Very Rev Richard William Herrick (3 December 1913 - 5 May 1981) was an eminent Anglican priest in the 20th century.

 He was educated at King Edward VI School Retford and Leeds University and was initially a civil servant.

 He was ordained after a period of study at the College of the Resurrection, Mirfield in 1940.

 he held curacies at Duston and Portsea, Portsmouth before being appointed Vicar of St Michael’s, Northampton in 1947, a post he held for a decade.

 He was then a Canon Residentiary of Chelmsford Cathedral until 1978 when he was appointed Provost of Chelmsford.

 He died in post.
</item 5>

<item 6>
Alice Walker: Beauty in Truth
Alice Walker: Beauty in Truth is a documentary film directed by Pratibha Parmar, made by Kali Films production company.

 The film follows the life of the Pulitzer Prize-winning author, poet and activist Alice Walker.

 Shooting began in May 2011.

 It was aired on the BBC on Monday July 8, 2013, and on PBS on February 7, 2014.
</item 6>

<item 7>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 7>

<item 8>
Warrior Marks
Warrior Marks: Female Genital Mutilation and the Sexual Blinding of Women is a 1993 book by Alice Walker with Pratibha Parmar, who made an award-winning
documentary of the same name.

 Following on from her 1992 novel "Possessing the Secret of Joy", Walker undertakes a journey to parts of Africa where clitoridectomy is still practised.

 "Warrior Marks" is a harrowing work as Walker interviews women who have had the operation done and finally interviews a woman—circumcised herself—who performs
the operation.
</item 8>

<item 9>
Alice R. Ballard
Alice R. Ballard.

 (born June 16, 1945, Florence, South Carolina) is an American ceramicist based in Greenville, South Carolina.

 Much of her work is characterized by the organic earthenware forms of closed containers, pinch pots, platters, pods, teapots, totems, small work, vessels and a
series she refers to as her white work.

 Much of her work is finished with terra sigillata.

 Alice considers her art to be “a reflection of [her] relationship with natural forms.

 It is often the metamorphosis of Nature's forms, as they change from season to season, that attracts her to that universal world in which differing life forms
share similar qualities."

 Her aesthetic heavily references the mother/child/germination metaphor and also explores the more evocative realm of wonder and awe.
</item 9>

<item 10>
Haniel Long
Haniel Clark Long (March 9, 1888 – October 17, 1956) was an American poet, novelist, publisher and academic.

 He is best known for his novella, "Interlinear to Cabeza de Vaca" (1936), a fictionalized account of the true story of a Spanish conquistador in 16th century
North America.
</item 10>
=== END ITEM id=1977258f623e422a ===

=== ITEM id=08d6bb4abd4b8e55 ===
Gold sentences and facts:
Sentence 0:  It is an annual event hosted by the Department of Urology at the University of Virginia, and the Charlottesville Track Club.
  Fact 0.0: The event is an annual event.
  Fact 0.1: The event is hosted by the Department of Urology at the University of Virginia.
  Fact 0.2: The event is hosted by the Charlottesville Track Club.
Sentence 1: Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.
  Fact 1.0: Panera Bread Company is an American chain.
  Fact 1.1: Panera Bread Company operates bakery-café fast casual restaurants.
  Fact 1.2: Panera Bread Company's restaurants are in the United States.
  Fact 1.3: Panera Bread Company's restaurants are in Canada.

Retrieved items:
<item 1>
Gino's Hamburgers
Gino's Hamburgers was a fast-food restaurant chain founded in Baltimore, Maryland, by Baltimore Colts defensive end Gino Marchetti and running back Alan Ameche,
along with their close friend Louis Fischer, in 1957.

 A new group of restaurants under the Gino's name involving some of the principals of the original chain was started in 2010.
</item 1>

<item 2>
List of fast food restaurant chains
The following is a list of notable current and former fast food restaurant chains, as distinct from fast casual restaurants (see List of casual dining
restaurant chains), coffeehouses (see List of coffeehouse chains), ice cream parlors (see List of ice cream parlor chains), and pizzerias (see List of pizza
chains).
</item 2>

<item 3>
Chipotle Mexican Grill
Chipotle Mexican Grill, Inc. ( ) is an American chain of fast casual restaurants in the United States, United Kingdom, Canada, Germany, and France, specializing
in tacos and Mission-style burritos. Its name derives from "chipotle", the Nahuatl name for a smoked and dried jalapeño chili pepper.

 The company trades on the New York Stock Exchange under the ticker symbol CMG.
</item 3>

<item 4>
Tudor's Biscuit World
Tudor's Biscuit World is a restaurant chain based in Huntington, West Virginia, most commonly found in West Virginia.

 Many West Virginia locations share a building with Gino's Pizza and Spaghetti, although the chain is more extensive than Gino's (which is exclusive to West
Virginia), having locations in southern Ohio, eastern Kentucky, and southwestern Virginia.

 In 2016 a franchise was opened in Panama City, Florida.

 Tudor's serves biscuits, biscuit sandwiches, homestyle breakfasts and dinners, muffins, and several side dishes.

 The chain was originally based in Charleston, West Virginia and many of the biscuit sandwiches are named for sports teams of interest in that area, including
teams at Marshall University, West Virginia University, and The University of Charleston.
</item 4>

<item 5>
Bill Steers Men's 4-Miler
The Bill Steers Men's 4-Miler (formerly Charlottesville Men's Four Miler) is the only all men's run in the Charlottesville area.

 The race was rededicated as a tribute to William D. Steers in 2015.

 It is an annual event hosted by the Department of Urology at the University of Virginia, and the Charlottesville Track Club.

 The event has also been sponsored by Panera Bread, BMW of Charlottesville, Starbucks, Pepsi and the City of Charlottesville.

 The City of Charlottesville also hosts the Women's Four Miler to benefit the UVA Cancer Center Breast Care Program.

 The event, held continuously since 2004, was conceived by William D. Steers with the purpose of raising awareness on men's health issues; to encourage men to
take control of their health, and become more physically active.
</item 5>

<item 6>
Panera Bread
Panera Bread Company is an American chain of bakery-café fast casual restaurants in the United States and Canada.

 Its headquarters are in Sunset Hills, Missouri, a suburb of St. Louis, and operates as Saint Louis Bread Company in the St. Louis metropolitan area.

 Offerings include soups, salads, pasta, sandwiches, specialty drinks, and bakery items.
</item 6>

<item 7>
William D. Steers
Dr. William D. Steers, August 19, 1955 – April 10, 2015, was a Paul Mellon professor and Chair of the Department of Urology at the School of Medicine of the
University of Virginia.

 Dr. Steers is past President of the American Board of Urology (ABU) and Editor of the "Journal of Urology".

 In 2003, the University of Virginia awarded Dr. Steers the Hovey Dabney Professorship.

 In 2004 Dr. Steers initiated the Charlottesville Men’s Four Miler road race in Virginia to raise funds for men’s health.

 Dr. Steers was a viticulturist aficionado, he co-owned Well Hung Vineyard in Charlottesville.

 He also authored YOURometer an iPhone app.

 used to record urological related symptoms.
</item 7>

<item 8>
Team Hole in the Wall
Team Hole in the Wall is an athletic fundraising initiative managed by the Hole in the Wall Gang Camp.

 Amateur athletes join up to raise a specified amount of funds and receive entrance to a marathon, bike ride or other athletic event in support of seriously ill
children served by The Hole in the Wall Gang Camp or one of its associated camps around the world.

 Established in 2005, Team Hole in the Wall offers entrance into more than two dozen marathons and cycling events, including the New York City Marathon, Boston
Marathon and AngelRide cycling event in Connecticut.

 New Canaan resident and Newman's Own Vice President of Marketing Michael “Mike” Havard ran in the ING New York City Marathon prior to 2004, but in that year he
decided to use his participation to raise funds for The Hole in the Wall Gang Camp, raising nearly $40,000.

 The next year, he served as captain of the first Team Hole in the Wall team of 40 runners, raising $150,000.

 The number of Team Hole in the Wall members and events has risen each year since.

 In 2010, more than 1,800 Team Hole in the Wall members are expected to participate in 20 athletic events.
</item 8>

<item 9>
Aladdin's Eatery
Aladdin's Eatery is a chain of franchised restaurants in the American Midwest and South-East, specializing in Lebanese cuisine.

 Adapted to American tastes, the sites are fast casual restaurants that also offer take out.
</item 9>

<item 10>
McAlister's Deli
McAlister's Deli is an American chain of fast casual restaurants founded in 1989 in Oxford, Mississippi by retired dentist Dr. Don Newcomb.

 There are currently over 400 locations in 26 states, ranging from Virginia in the East to Florida in the South to Arizona in West to Michigan in the North.

 The menu includes deli sandwiches, "Texas-size" spuds (baked potatoes), soups, salads, and desserts, as well as catering items such as sandwich trays and boxed
lunches.

 The chain is also known for its McAlister's Famous Sweet Tea, which is available by the glass or by the gallon.
</item 10>
=== END ITEM id=08d6bb4abd4b8e55 ===

=== ITEM id=009f85f8364aca9e ===
Gold sentences and facts:
Sentence 0: Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11, 1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical physics.
  Fact 0.0: Nikolay Mitrofanovich Krylov was born on 29 November [O.S. 17 November] 1879.
  Fact 0.1: Nikolay Mitrofanovich Krylov died on May 11, 1955.
  Fact 0.2: Nikolay Mitrofanovich Krylov's Russian name is Николай Митрофанович Крылов.
  Fact 0.3: Nikolay Mitrofanovich Krylov's Ukrainian name is Микола Митрофанович Крилов.
  Fact 0.4: Nikolay Mitrofanovich Krylov was a Russian mathematician.
  Fact 0.5: Nikolay Mitrofanovich Krylov was a Soviet mathematician.
  Fact 0.6: Nikolay Mitrofanovich Krylov is known for works on interpolation.
  Fact 0.7: Nikolay Mitrofanovich Krylov is known for works on non-linear mechanics.
  Fact 0.8: Nikolay Mitrofanovich Krylov is known for works on numerical methods for solving equations of mathematical physics.
Sentence 1: Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.
  Fact 1.0: Anatoly Timofeevich Fomenko's Russian name is Анатолий Тимофеевич Фоменко.
  Fact 1.1: Anatoly Timofeevich Fomenko was born on 13 March 1945 in Stalino, USSR.
  Fact 1.2: Anatoly Timofeevich Fomenko is a Soviet mathematician.
  Fact 1.3: Anatoly Timofeevich Fomenko is a Russian mathematician.
  Fact 1.4: Anatoly Timofeevich Fomenko is a professor at Moscow State University.
  Fact 1.5: Anatoly Timofeevich Fomenko is well known as a topologist.
  Fact 1.6: Anatoly Timofeevich Fomenko is a member of the Russian Academy of Sciences.

Retrieved items:
<item 1>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 1>

<item 2>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 2>

<item 3>
Ural Federal University
The Ural Federal University named after the first President of Russia B. N. Yeltsin (Russian: Уральский федеральный университет имени первого Президента России
Б.Н.

 Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural
State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.

 Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.

 Training of students is carried out in four main areas of knowledge and 108 academic majors.

 The links between the university and intermediate educational institutions are actively developed.
</item 3>

<item 4>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 4>

<item 5>
Describing function
In control systems theory, the describing function (DF) method, developed by Nikolay Mitrofanovich Krylov and Nikolay Bogoliubov in the 1930s, and extended by
Ralph Kochenburger is an approximate procedure for analyzing certain nonlinear control problems.

 It is based on quasi-linearization, which is the approximation of the non-linear system under investigation by a linear time-invariant (LTI) transfer function
that depends on the amplitude of the input waveform.

 By definition, a transfer function of a true LTI system cannot depend on the amplitude of the input function because an LTI system is linear.

 Thus, this dependence on amplitude generates a family of linear systems that are combined in an attempt to capture salient features of the non-linear system
behavior.

 The describing function is one of the few widely applicable methods for designing nonlinear systems, and is very widely used as a standard mathematical tool
for analyzing limit cycles in closed-loop controllers, such as industrial process controls, servomechanisms, and electronic oscillators.
</item 5>

<item 6>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 6>

<item 7>
New Chronology (Fomenko)
The New Chronology is a pseudohistorical theory which argues that the conventional chronology of Middle Eastern and European history is fundamentally flawed,
and that events attributed to the civilizations of the Roman Empire, Ancient Greece and Ancient Egypt actually occurred during the Middle Ages, more than a
thousand years later.

 The central concepts of the New Chronology are derived from the ideas of Russian scholar Nikolai Morozov (1854–1946), although work by French scholar Jean
Hardouin (1646–1729) can be viewed as an earlier predecessor.

 However, the New Chronology is most commonly associated with Russian mathematician Anatoly Fomenko (born 1945), although published works on the subject are
actually a collaboration between Fomenko and several other mathematicians.

 The concept is most fully explained in "History: Fiction or Science?"

, originally published in Russian.
</item 7>

<item 8>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 8>

<item 9>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 9>

<item 10>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 10>
=== END ITEM id=009f85f8364aca9e ===

=== ITEM id=1000c029c172d87b ===
Gold sentences and facts:
Sentence 0:  His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed according to its experimental approach.
  Fact 0.0: His experimental works are well known in the Malayalam Cinema industry.
  Fact 0.1: His experimental works are appropriately discussed in the Malayalam Cinema industry.
  Fact 0.2: His debut movie is Jalachhayam.
  Fact 0.3: Jalachhayam was released in 2010.
  Fact 0.4: Jalachhayam was well discussed according to its experimental approach.
Sentence 1: Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA, Britain and Germany.
  Fact 1.0: Jacques Feyder was born on 21 July 1885.
  Fact 1.1: Jacques Feyder died on 24 May 1948.
  Fact 1.2: Jacques Feyder was a Belgian actor.
  Fact 1.3: Jacques Feyder was a Belgian screenwriter.
  Fact 1.4: Jacques Feyder was a Belgian film director.
  Fact 1.5: Jacques Feyder worked principally in France.
  Fact 1.6: Jacques Feyder also worked in the USA.
  Fact 1.7: Jacques Feyder also worked in Britain.
  Fact 1.8: Jacques Feyder also worked in Germany.

Retrieved items:
<item 1>
Sathish Kalathil
Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam.

 He is also Story Writer, and Lyricist.

 His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed
according to its experimental approach.

 Actually, his first venture was in 2008 named Veena Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in movie.

 In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about Municipal solid waste.
</item 1>

<item 2>
Bandi Sailu
Bandi Sailu (1928 – 15 July 2000) or Sayilu Saayilu was a Marxist revolutionary born in a Dalit family in Mandapelli village, Duggondi mandal, Narsampeta
Thalukh in the Warangal district of Andhra Pradesh, India.
</item 2>

<item 3>
Graph query row:
person: Sathish Kalathil
birth_date: None
</item 3>

<item 4>
Jacques Feyder
Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA,
Britain and Germany.

 He was a leading director of silent films during the 1920s, and in the 1930s he became associated with the style of poetic realism in French cinema.

 He adopted French nationality in 1928.
</item 4>

<item 5>
Jalachhayam
Jalachhayam (മലയാളം: ) is a 2010 Indian Malayalam-language Experimental film, produced and directed by Sathish Kalathil under the banner of 'The People's
Films'.

 The story was written by Sujith Aalungal and describes the warm relationship between a village man and an artist (painter) from the city.
</item 5>

<item 6>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 6>

<item 7>
T. Muthuraj
T. MuthuRaj (born July 29, 1969, Coimbatore, India) is an Indian film production designer.

 He entered the film industry after completing his B.Sc.

 degree in fine arts as first assistant to Sabu Cyril after which he has served as production designer in critically acclaimed and celebrated films in major
Indian languages.

 His big break came in the 1997 Malayalam film "Guru" starring Mohanlal and Suresh Gopi, directed by Rajiv Anchal, for which he received the Kerala State Film
Award for Best Art Director.

 Muthuraj has since worked in such path breaking hits as "Arputha Theevu", "Angaditheru", "Pazhasi Raja", "Irumbukottai Murattu Singham" and "Avan Ivan".

 He has contributed to the success of 43 feature films and 185 commercials.

 He has won many film awards for best art direction and for his creative works.

 His latest work is "Nanban" (2012) directed by Shankar, which has won huge appreciation from all quarters.
</item 7>

<item 8>
Kerala School Kalolsavam
School Arts Festival of Kerala (Malayalam: കേരള സ്കൂൾ കലോത്സവം) is an annual event conducted by the state government of Kerala, featuring several art
competitions for high school and higher secondary school students of Kerala.

 The festival was started in 1956, and till 2008, it was called as "Kerala State School Youth Festival".

 The participants are students from classes 8th to 12th.

 Winners from different revenue districts for a particular event will be competing in state level competition.

 The event is usually conducted in December–January months of a year and is considered to be the biggest cultural event of Asia.

 57th edition of Kerala School Kalolsavam 2017 is scheduled for January 16 to 22 at Kannur.
</item 8>

<item 9>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 9>

<item 10>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 10>
=== END ITEM id=1000c029c172d87b ===

=== ITEM id=71047d281c695f37 ===
Gold sentences and facts:
Sentence 0: The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.
  Fact 0.0: The Bishop of Ely is the ordinary of the Diocese of Ely.
  Fact 0.1: The Diocese of Ely is a Church of England diocese.
  Fact 0.2: The Diocese of Ely is in the Province of Canterbury.
Sentence 1:  Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
  Fact 1.0: Conway became Bishop of Ely in 2010.
  Fact 1.1: Conway was translated to Bishop of Ely from the Diocese of Salisbury.
  Fact 1.2: Conway was Bishop suffragan of Ramsbury before becoming Bishop of Ely.
  Fact 1.3: Conway was Bishop suffragan of Ramsbury in the Diocese of Salisbury.
Sentence 2: The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.
  Fact 2.0: The Diocese of Salisbury is a Church of England diocese.
  Fact 2.1: The Diocese of Salisbury is in the south of England.
  Fact 2.2: The Diocese of Salisbury is within the ecclesiastical Province of Canterbury.

Retrieved items:
<item 1>
Diocese of Salisbury
The Diocese of Salisbury is a Church of England diocese in the south of England, within the ecclesiastical Province of Canterbury.

 The diocese covers most of Dorset (excepting the deaneries of Bournemouth and Christchurch, which fall within the Diocese of Winchester), and most of Wiltshire
(excepting a part in the north and Swindon).

 The diocese is led by the Bishop of Salisbury (Nick Holtam) and the diocesan synod.

 The bishop's seat is at Salisbury Cathedral.
</item 1>

<item 2>
Bishop of Ely
The Bishop of Ely is the ordinary of the Church of England Diocese of Ely in the Province of Canterbury.

 The diocese roughly covers the county of Cambridgeshire (with the exception of the Soke of Peterborough), together with a section of north-west Norfolk and has
its episcopal see in the City of Ely, Cambridgeshire, where the seat is located at the Cathedral Church of the Holy Trinity.

 The current bishop is Stephen Conway, who signs "+Stephen Elien:" (abbreviation of the Latin adjective "Eliensis", meaning "of Ely").

 The diocesan bishops now reside in Bishop's House, Ely, the former cathedral deanery.

 Conway became Bishop of Ely in 2010, translated from the Diocese of Salisbury where he was Bishop suffragan of Ramsbury.
</item 2>

<item 3>
Roman Catholic Ecclesiastical Province of Miami
The Ecclesiastical Province of Miami is a Catholic ecclesiastical province covering the U.S. state of Florida.

 Its metropolitan bishop is the Archbishop of Miami, head of the Archdiocese of Miami.

 The province additionally includes the suffragan dioceses of Orlando, Palm Beach, Pensacola-Tallahassee, St. Augustine, St. Petersburg, and Venice.
</item 3>

<item 4>
Graph query row:
diocese: Diocese of Ely
ecclesiastical_province: Province of Canterbury
</item 4>

<item 5>
Ecclesiastical Province of Ontario
The Ecclesiastical Province of Ontario is one of the Anglican Church of Canada's four ecclesiastical provinces.

 It was established in 1912 out of six dioceses of the Ecclesiastical Province of Canada located in the civil Province of Ontario, and the Diocese of Moosonee
from the Ecclesiastical Province of Rupert's Land.
</item 5>

<item 6>
Dean of Chelmsford
The Dean of Chelmsford is the head ("primus inter pares" – first among equals) and chair of the Cathedral Chapter, the governing body of Chelmsford Cathedral,
the Cathedral Church of St Mary the Virgin, St Peter and St Cedd.

 Before 2000 the post was designated as a provost, which was then the equivalent of a dean at most English cathedrals.

 The cathedral is the mother church of the Diocese of Chelmsford and seat of the Bishop of Chelmsford.

 The Dean of Chelmsford is also responsible for the Chapel of St Peter-on-the-Wall at Bradwell-on-Sea, founded by St Cedd, among the oldest church buildings in
regular use in England.
</item 6>

<item 7>
Diocese of Guildford
The Diocese of Guildford is a Church of England diocese covering nine of the eleven districts in Surrey, much of north-east Hampshire and a parish in Greater
London.

 The cathedral is Guildford Cathedral and the bishop is the Bishop of Guildford.

 Of the two provinces of the church it falls within the Province of Canterbury.
</item 7>

<item 8>
Dean of Guildford
The Dean of Guildford is the head ("primus inter pares" – first among equals) and chair of the chapter of canons, the ruling body of Guildford Cathedral.

 The dean and chapter are based at the "Cathedral Church of the Holy Spirit" in Guildford.

 The cathedral is the mother church of the Diocese of Guildford and seat of the Bishop of Guildford.

 The current dean is Dianna Gwilliams, who was installed on 15 September 2013.
</item 8>

<item 9>
Ecclesiastical Province of Freiburg
The Ecclesiastical Province of Freiburg ("Kirchenprovinz Freiburg") or Upper Rhenish Ecclesiastical Province ("Oberrheinische Kirchenprovinz") is an
ecclesiastical province of the Roman Catholic Church in the Upper Rhine area of Germany, centring on Freiburg im Breisgau.

 It covers the Roman Catholic Archdiocese of Freiburg, the Roman Catholic Diocese of Mainz and the Roman Catholic Diocese of Rottenburg-Stuttgart, covering
large areas of Baden-Württemberg and Hesse and small parts of Rhineland-Palatinate.

 Its metropolitan bishop is the Archbishop of Freiburg - that Archdiocese and the Province were both set up in 1821 in the wake of the 1801 Concordat and the
1815 Congress of Vienna.
</item 9>

<item 10>
Ecclesiastical province
An ecclesiastical province is a general term for one of the basic forms of jurisdiction in Christian Churches with traditional hierarchical structure, including
Western Christianity and Eastern Christianity.

 In general, ecclesiastical province is consisted of several dioceses (or eparchies), one of them being the archdiocese (or archeparchy), headed by metropolitan
bishop or archbishop who has ecclesiastical jurisdiction over all other bishops of the province.
</item 10>
=== END ITEM id=71047d281c695f37 ===
