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

This file has 6 items.

=== ITEM id=3aa92492e72b2339 ===
Gold sentences and facts:
Sentence 0: John James McCoy is a fictional character in the television drama Law & Order . He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order , four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the made - for - TV movie Exiled .
  Fact 0.0: John James McCoy is a fictional character in the television drama Law & Order.
  Fact 0.1: John James McCoy was created by Michael S. Chernuchin.
  Fact 0.2: John James McCoy was portrayed by Sam Waterston from 1994 until the end of the series in 2010.
  Fact 0.3: John James McCoy is the second-longest tenured character on the show, with 16 seasons.
  Fact 0.4: John James McCoy's tenure ranks just after Lt. Anita Van Buren, who had 17 seasons.
  Fact 0.5: Lt. Anita Van Buren was portrayed by S. Epatha Merkerson.
  Fact 0.6: John James McCoy appeared in 368 episodes of Law & Order.
  Fact 0.7: John James McCoy appeared in four episodes of Law & Order: Special Victims Unit.
  Fact 0.8: John James McCoy appeared in two episodes of Law & Order: Trial by Jury.
  Fact 0.9: John James McCoy appeared in two episodes of Homicide: Life on the Street.
  Fact 0.10: John James McCoy appeared in the made-for-TV movie Exiled.
Sentence 1: Jack McCoy brings 24 years of experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends -- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick , and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics occasionally incur negative publicity for the DA 's office . His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial .
  Fact 1.0: Jack McCoy brings 24 years of experience with him.
  Fact 1.1: Jack McCoy is appointed executive assistant district attorney by Adam Schiff.
  Fact 1.2: Adam Schiff is played by Steven Hill.
  Fact 1.3: Jack McCoy is appointed in the season-five premiere episode 'Second Opinion'.
  Fact 1.4: Jack McCoy quickly establishes himself as a more unconventional, ruthless litigator than his predecessor.
  Fact 1.5: Jack McCoy's predecessor is Ben Stone.
  Fact 1.6: Ben Stone is played by Michael Moriarty.
  Fact 1.7: Jack McCoy often bends trial rules to get convictions.
  Fact 1.8: Jack McCoy sometimes breaks trial rules to get convictions.
  Fact 1.9: Jack McCoy finds tenuous rationales for charging defendants with crimes when the original charges fail to stick.
  Fact 1.10: Jack McCoy charges innocent people to frighten them into testifying against others.
  Fact 1.11: Jack McCoy is found in contempt of court 80 times for such behavior.
  Fact 1.12: Jack McCoy's tactics occasionally incur negative publicity for the DA's office.
  Fact 1.13: Jack McCoy's underlying motivation is not corruption, he maintains.
  Fact 1.14: Jack McCoy's underlying motivation is a sincere desire to see justice done.
  Fact 1.15: Jack McCoy has gone after defendants accused of perverting the justice system to arrange wrongful convictions.
  Fact 1.16: Jack McCoy pursues those defendants with just as much determination as his more mundane cases.
  Fact 1.17: Jack McCoy's aggressive actions in the courts have earned him the nickname 'Hang 'em High McCoy'.
  Fact 1.18: Jack McCoy has developed a reputation with both colleagues and rival attorneys.
  Fact 1.19: Jack McCoy was once referred to as 'the top of the legal food chain' by a rival attorney during a trial.
Sentence 2: `` Entitled '' ( onscreen ) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) .
  Fact 2.0: The episode 'Entitled' (onscreen) is where McCoy makes his first appearance on the series.
  Fact 2.1: McCoy works with his assistant, Abbie Carmichael.
  Fact 2.2: Abbie Carmichael is played by Angie Harmon.
  Fact 2.3: McCoy assists the Special Victims Unit in solving a cold case.
  Fact 2.4: Briscoe handled the cold case years ago.
  Fact 2.5: Briscoe handled the cold case with his former partner, Detective Mike Logan.
  Fact 2.6: Detective Mike Logan is played by Chris Noth.

Retrieved items:
<item 1>
Jack McCoy
He was created by Michael S. Chernuchin and portrayed by Sam Waterston from 1994 until the end of the series in 2010 . He is the second - longest tenured
character on the show ( 16 seasons ) , after Lt. Anita Van Buren ( 17 seasons ; portrayed by S. Epatha Merkerson ) . He appeared in 368 episodes of Law & Order
, four episodes of Law & Order : Special Victims Unit , two episodes of Law & Order : Trial by Jury , two episodes of Homicide : Life on the Street , and the
made - for - TV movie Exiled . Waterston 's performance as McCoy on the New York - based series was so popular that it resulted in him being declared a ``
Living Landmark '' by the New York Landmarks Conservancy , along with fellow longtime series cast member Jerry Orbach ( who portrayed the popular police
detective Lennie Briscoe for 12 years ) . Contents ( hide ) 1 Character overview 2 Personal life 3 Notable conflicts 4 In Law & Order : Special Victims Unit 4.1
As executive assistant district attorney 4.1.
</item 1>

<item 2>
Jack McCoy
Jack McCoy - wikipedia Jack McCoy Jump to : navigation , search For the politician and labor activist , see Jack E. McCoy . Jack McCoy Law & Order character
First appearance `` Second Opinion '' `` Entitled '' ( SVU ) Last appearance `` Rubber Room '' ( Law & Order ) `` The Undiscovered Country '' ( SVU ) Portrayed
by Sam Waterston Time on show 1994 -- 2010 , 2018 Seasons 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 Credited appearances 368
episodes ( L&O ) 2 episodes ( HLOTS ) 4 episodes ( SVU ) 2 episodes ( TBJ ) 376 episodes ( total ) Preceded by Ben Stone ( EADA ) Arthur Branch ( DA ) ( L&O )
Succeeded by Michael Cutter ( EADA ) Information Family Rebecca McCoy ( daughter ) John James McCoy is a fictional character in the television drama Law & Order
.
</item 2>

<item 3>
Jack McCoy
wikipedia.org/w/index.php?title=Jack_McCoy&oldid=838021970 '' Categories : Law & Order characters Fictional lawyers Fictional American people of Irish descent
Fictional characters introduced in 1994 Fictional district attorneys Fictional victims of child abuse Crossover characters in television Hidden categories :
Pages using deprecated image syntax Articles using Infobox character with multiple unlabeled fields Talk Contents About Wikipedia Deutsch Hrvatski 日本 語
Português Srpskohrvatski / српскохрватски Edit links This page was last edited on 24 April 2018 , at 13 : 14 . About Wikipedia
</item 3>

<item 4>
Jack McCoy
Although McCoy insists that he is a prosecutor , and not a politician , McCoy replaces Branch as the DA immediately thereafter . In Law & Order : Special
Victims Unit ( edit ) McCoy has appeared in four episodes of Law & Order : Special Victims Unit ; however , he is often referred to in the series , and his
actions affect the ADAs working with the Special Victims Unit . As executive assistant district attorney ( edit ) Season one ( edit ) `` Entitled '' ( onscreen
) : McCoy makes his first appearance on the series . McCoy , working with his assistant , Abbie Carmichael ( Angie Harmon ) , assists the Special Victims Unit
in solving a cold case , that Briscoe handled years ago with his former partner , Detective Mike Logan ( Chris Noth ) . Season two ( edit ) `` Baby Killer '' (
mention ) : After a murder intertwines with ADA Alexandra Cabot 's ( Stephanie March ) sex - crimes case , Chief Assistant District Attorney Charlie Phillips (
Jeffrey DeMunn ) , Lewin 's second in command , hands the murder case to McCoy . As district attorney ( edit ) Season 9 ( edit ) `` Blinded '' ( onscreen ) :
McCoy had been promoted to district attorney .
</item 4>

<item 5>
Jack McCoy
1 Season one 4.1. 2 Season two 4.2 As district attorney 4.2. 1 Season 9 4.2. 2 Season 10 4.2. 3 Season 11 4.2. 4 Season 12 4.2. 5 Season 13 4.2. 6 Season 19 5
Reception 6 District attorney 's office timeline 7 Appearances on other TV shows 8 References Character overview ( edit ) Jack McCoy brings 24 years of
experience with him as he is appointed executive assistant district attorney by Adam Schiff ( Steven Hill ) in the season - five premiere episode `` Second
Opinion '' . He quickly establishes himself as a more unconventional , ruthless litigator than his predecessor , Ben Stone ( Michael Moriarty ) . He often bends
-- and sometimes breaks -- trial rules to get convictions , finds tenuous rationales for charging defendants with crimes when the original charges fail to stick
, and charges innocent people to frighten them into testifying against others . McCoy is found in contempt of court 80 times for such behavior , and his tactics
occasionally incur negative publicity for the DA 's office .
</item 5>

<item 6>
Jack McCoy
( hide ) Law & Order Characters Police Sgt . Max Greevey Sgt . Phil Cerreta Lennie Briscoe Joe Fontana Ed Green Cyrus Lupo Mike Logan Rey Curtis Nick Falco Nina
Cassady Kevin Bernard Captain Donald Cragen Lieutenant Anita Van Buren District attorneys Benjamin Stone Jack McCoy Michael Cutter Paul Robinette Claire Kincaid
Jamie Ross Abbie Carmichael Serena Southerlyn Alexandra Borgia Connie Rubirosa Adam Schiff Nora Lewin Arthur Branch Recurring characters Elizabeth Olivet , Ph.
D . Emil Skoda , M.D. Elizabeth Rodgers , M.D.
</item 6>

<item 7>
Jack McCoy
By the end of the 18th season , however , McCoy discovers that Shalvoy is involved in a prostitution scandal that is tied to a murder case he is prosecuting .
Angered and disappointed , McCoy orders Cutter to start investigating Shalvoy , who retaliates by lending his support to McCoy 's opponent in the election . In
the last episode of season 19 , `` The Drowned and the Saved '' , Cutter uncovers evidence that Shalvoy tried to buy a Senate seat for his wife , and threatens
him with public disgrace unless he resigns . McCoy 's opponent suddenly has no patron , giving McCoy good prospects for victory . In the opening episode of
season 20 , McCoy was revealed to have won the election ; he serves as DA for the remainder of this final season of Law & Order . Personal Life ( edit ) While
he is a brilliant legal mind , McCoy has more than a few personal demons . He was abused by his father , an Irish Chicago policeman who had also beat Jack 's
mother , and who eventually died of cancer . McCoy says that his determination and unyielding work ethic are a byproduct of having been harshly punished by his
father for losing at anything .
</item 7>

<item 8>
Jack McCoy
District attorney 's office timeline ( edit ) Time period Executive assistant district attorney ( EADA ) Assistant district attorney ( ADA ) District attorney (
DA ) 1994 -- 1996 Jack McCoy Claire Kincaid Adam Schiff 1996 -- 1998 Jamie Ross 1998 -- 2000 Abbie Carmichael 2000 -- 2001 Nora Lewin 2001 -- 2002 Serena
Southerlyn 2002 -- 2005 Arthur Branch 2005 -- 2006 Alexandra Borgia 2006 -- 2007 Connie Rubirosa 2008 -- 2010 Michael Cutter Jack McCoy Appearances on other TV
shows ( edit ) Homicide : Life on the Street Season 6 Episode 5 : `` Baby ,
</item 8>

<item 9>
Jack McCoy
His underlying motivation , however , is not , he maintains , corruption , but a sincere desire to see justice done . To that end , McCoy has gone after
defendants accused of perverting the justice system to arrange wrongful convictions with just as much determination as his more mundane cases . Such aggressive
actions in the courts have earned him the nickname `` Hang ' em High McCoy '' . He has subsequently developed a reputation with both colleagues and rival
attorneys , once being referred to as `` the top of the legal food chain '' by a rival attorney during a trial . Following the 17th season ( 2006 -- 2007 ) ,
Jack McCoy was appointed interim district attorney , taking over from Arthur Branch ( Fred Thompson ) . McCoy 's appearance on Law & Order : Special Victims
Unit on the November 13 , 2007 , episode `` Blinded '' , marked his first appearance in the Law & Order universe as district attorney . The replacement for his
former position is Michael Cutter ( Linus Roache ) , a prosecutor with a penchant for recklessness not unlike McCoy 's own in his younger days . This
occasionally presents political difficulties for the new district attorney .
</item 9>

<item 10>
Jack McCoy
Reception ( edit ) Entertainment Weekly television critic Ken Tucker has praised Law & Order 's creator Dick Wolf for putting McCoy at the center of `` some of
the best episodes of the immortal series ' 19th season . '' Tucker elaborates how the character , riding `` herd over a couple of stubborn young bucks --
assistant DAs Mike Cutter ( Linus Roache ) and Connie Rubirosa ( Alana de la Garza ) -- McCoy argues , bellows orders , and croaks with outrage when his charges
disobey his legal advice . ''
</item 10>
=== END ITEM id=3aa92492e72b2339 ===

=== ITEM id=b2adb5e095de6ecc ===
Gold sentences and facts:
Sentence 0: `` Like a Boss '' is a song written and recorded by comedy hip hop troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean version that was produced in both audio and video . In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work .
  Fact 0.0: Like a Boss is a song.
  Fact 0.1: Like a Boss was written and recorded by the comedy hip hop troupe The Lonely Island.
  Fact 0.2: The Lonely Island recorded Like a Boss for their debut studio album Incredibad.
  Fact 0.3: Like a Boss is a parody of the first official single from Slim Thug's first album Already Platinum.
  Fact 0.4: The first official single from Slim Thug's first album Already Platinum is also named Like a Boss.
  Fact 0.5: Like a Boss had multiple versions.
  Fact 0.6: One version of Like a Boss was audio-only and did not have the celebrity interviewer.
  Fact 0.7: Like a Boss had a clean version that was produced in both audio and video.
  Fact 0.8: In 2011 Like a Boss was subject to a lawsuit.
  Fact 0.9: The 2011 lawsuit over Like a Boss was from the producers of the song.
  Fact 0.10: The producers of Like a Boss claimed they never received proper compensation for their work.

Retrieved items:
<item 1>
Like a Boss
Unsourced material may be challenged and removed . ( February 2011 ) ( Learn how and when to remove this template message ) ( Learn how and when to remove this
template message ) `` Like a Boss '' Song by The Lonely Island featuring Seth Rogen from the album Incredibad Recorded 2008 Genre Comedy hip hop Length 1 : 46
Label Universal Republic Records Songwriter ( s ) Andy Samberg , Akiva Schaffer , Jorma Taccone , A. Banks Producer ( s ) Rick tha Rular Incredibad track
listing `` Shrooms ( skit ) '' ( 9 ) `` Like a Boss '' ( 10 ) `` We Like Sportz '' ( 11 ) `` Like a Boss '' is a song written and recorded by comedy hip hop
troupe The Lonely Island for their debut studio album Incredibad . The song is a parody of the first official single from Slim Thug 's first album Already
Platinum , also named `` Like a Boss '' . The song had multiple versions , including an audio - only that did not have the celebrity interviewer and a clean
version that was produced in both audio and video .
</item 1>

<item 2>
Like a Boss
TV Guide.com . Retrieved 23 June 2014 . Jump up ^ Kaufman , Gill . `` ' SNL , ' Andy Samberg Sued By Producers over ' Like A Boss ' - MTV '' . MTV News .
Retrieved 11 July 2014 . External links ( edit ) Beard , Lanford ( November 8 , 2011 ) . `` NBC , Andy Samberg sued for copyright infringement '' .
Entertainment Weekly . Retrieved November 8 , 2011 . Original video : `` Like a Boss '' on YouTube Censored video : `` Like a Boss - Clean Version '' on YouTube
Video and Song Lyrics : `` Like a Boss '' Lonely Island 's Song Page : `` Like a Boss '' ( hide ) The Lonely Island Andy Samberg Akiva Schaffer Jorma Taccone
Discography Studio albums Incredibad Turtleneck & Chain The Wack Album Popstar : Never Stop Never Stopping Singles `` Dick in a Box '' `` Jizz in My Pants '' ``
I 'm on a Boat '' `` I Just Had Sex '' `` The Creep '' `` We 're Back ! ''
</item 2>

<item 3>
Like a Boss
In 2011 it was also subject to a lawsuit from the producers of the song who claimed they never received proper compensation for their work . Contents ( hide ) 1
Song 2 Video 3 Popularity 4 Lawsuit 5 Censorship 6 References 7 External links Song ( edit ) `` Like a Boss '' premiered as a Saturday Night Live Digital Short
on April 4 , 2009 . In the song , Andy Samberg is a businessman , evidently in middle management , having a performance review . The reviewer asks him to
describe the things he does in an average day , and does so for the remainder of the song in his trademark aggressive style while repeating the lyric `` like a
boss '' after every activity . He first lists activities which are fairly mundane , such as communicating to bigger bosses in corporate headquarters , approving
memorandums , remembering birthdays , leading a workshop , and promoting synergy . However , the song takes a foul turn after he mentions his female subordinate
Deborah , whom he repeatedly pursues , rejecting him , causing him to `` swallow sadness '' , face a sexual harassment lawsuit and be passed up for a promotion
.
</item 3>

<item 4>
Like a Boss
Like a Boss - wikipedia Like a Boss Jump to : navigation , search ( hide ) This article has multiple issues . Please help improve it or discuss these issues on
the talk page . ( Learn how and when to remove these template messages ) This article 's lead section does not adequately summarize key points of its contents .
Please consider expanding the lead to provide an accessible overview of all important aspects of the article . Please discuss this issue on the article 's talk
page . ( January 2015 ) This article needs additional citations for verification . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed .
</item 4>

<item 5>
Like a Boss
`` Jack Sparrow '' `` 3 - Way ( The Golden Rule ) '' `` YOLO '' `` Spring Break Anthem '' `` Hugs '' Other songs `` Lazy Sunday '' `` Like a Boss '' `` Cool
Guys Do n't Look at Explosions '' `` Everything Is Awesome '' Film Hot Rod ( 2007 ) Popstar : Never Stop Never Stopping ( 2016 ) Television SNL Digital Shorts
Party Over Here Book Retrieved from `` https://en.wikipedia.org/w/index.php?title=Like_a_Boss&oldid=816720124 '' Categories : The Lonely Island songs 2009
singles Saturday Night Live sketches Comedy rap songs Hidden categories : Wikipedia introduction cleanup from January 2015 All pages needing cleanup Articles
covered by WikiProject Wikify from January 2015 All articles covered by WikiProject Wikify Articles needing additional references from February 2011 All
articles needing additional references Articles with multiple maintenance issues Articles with hAudio microformats Articles needing additional references from
January 2015 Talk Contents About Wikipedia Nederlands Simple English Edit links This page was last edited on 23 December 2017 , at 06 : 20 . Text is available
under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
</item 5>

<item 6>
Like a Boss
Abby Elliott and Bobby Moynihan can be seen in the video as background dancers . Popularity ( edit ) Prior to the release of the video on Saturday Night Live ,
the term `` Like a Boss '' was not often searched for or used in social context . Though it was indeed searched for prior to the release of the video , the term
clearly spiked at release and peaked on Google in April 2009 , hitting a 100 on the Google Trend chart . It has also become popular on social media sites such
as Twitter Tumblr , and Facebook , where it is a popular hashtag to cite a task completed , goal achieved or successful action that someone is proud of doing .
It has also led to parody videos such as `` Like a Bus '' , where a bus is shown flying through traffic and often bad weather conditions with no regard or
slowing for the bad conditions . On 22 October 2012 , `` Like A Boss '' surpassed 100 million views . The video was featured by The Huffington Post , and TV
Guide . Lawsuit ( edit ) In November 2011 a production team from St. Louis filed charges against Andy Samberg and Co. , NBC Universal and Universal Republic
Records .
</item 6>

<item 7>
Like a Boss
Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use
and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 7>

<item 8>
Like a Boss
Banks , however , claims that he has yet to see any sort of a royalty for the song . Another track from the duo was later featured on the band 's 2011 album
Turtleneck & Chain in the song `` Shy Ronnie 2 : Ronnie & Clyde . '' Censorship ( edit ) This section does not cite any sources . Please help improve this
section by adding citations to reliable sources . Unsourced material may be challenged and removed . ( January 2015 ) ( Learn how and when to remove this
template message ) On the NBC version of this digital short , the part where Andy has a gun in his mouth and whines , `` Oh fuck , man , I ca n't fuckin ' do it
. Shit ! '' had the shot of Andy with the gun in his mouth shortened and redubbed the line to , `` Oh crap , man , I ca n't effin ' do it . Ugh ! ! ''
References ( edit ) Jump up ^ Van Dril , Eric ( 5 April 2009 ) . `` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' .
</item 8>

<item 9>
Like a Boss
`` ' SNL ' Digital Short : ' Like A Boss ' ( With Video ) '' . The Ripcord . Archived from the original on 11 April 2009 . Retrieved 2014 - 06 - 21 . Jump up ^
`` Google Trends - websearch interest : like a boss - Worldwide , 2004 - present '' . Google Trends . Jump up ^ `` Twitter / search - # likeaboss '' .
Twitter.com . Jump up ^ `` like a boss on Tumblr '' . Tumblr.com . Jump up ^ `` # likeaboss '' . Facebook.com . Jump up ^ `` Like a BUS ! Playlist '' .
Youtube.com . Jump up ^ `` Like A Boss : SNL Digital Short ( VIDEO ) '' . The Huffington Post . 6 May 2009 . Retrieved 2014 - 06 - 23 . Jump up ^ DiNunno , Gina
. `` Today 's News : Our Take - Seth Rogen and Andy Samberg in SNL Digital Short `` Like a Boss '' `` . TV Guide.com . Retrieved 23 June 2014 .
</item 9>

<item 10>
Like a Boss
The video ends with the giant fish delivering the final words `` like a boss . '' Video ( edit ) The video features Seth Rogen , recording over the vocals done
by Akiva Schaffer on the single. The video adds additional visuals not included with the words of the song . It ends with Samberg 's tombstone being shown (
which has the epitaph `` He was da Best '' ) , and a skeptical Rogen trying to confirm some of the more bizarre events with Samberg . The `` bomb the Russians
'' part shows Samberg attacking St. Basil 's Cathedral . The video features brief cameo appearances from Saturday Night Live writer John Mulaney as the worker
that Samberg micromanages , Casey Wilson as the woman giving Samberg the memo , Kristen Wiig as Samberg 's office crush Debra , Jason Sudeikis as Samberg 's
boss and Bill Hader as the man on whom Samberg performs fellatio . It also features Jorma Taccone as the man giving him the lawsuit , and Akiva Schaffer selling
him the gun . Abby Elliott and Bobby Moynihan can be seen in the video as background dancers .
</item 10>
=== END ITEM id=b2adb5e095de6ecc ===

=== ITEM id=d97961b24c7512a2 ===
Gold sentences and facts:
Sentence 0: Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher Usmanov ( 30.04 % ) 15 ! $22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M $3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 ! $2.3 B Steel and energy Theatre production Planet Hollywood Ocean Software Huddersfield Town Dean Hoyle 09 ! $475 M Card Factory Leicester City The Srivaddhanaprabha Family 10 ! $4.9 B King Power International Group Liverpool ( more information ) John W. Henry 11 ! $2.5 B Fenway Sports Group Manchester City ( more information ) Mansour bin Zayed Al Nahyan ( 87 % ) China Media Capital ( 13 % ) 01 ! $44 B Oil and industry investment Government Manchester United ( more information ) The Glazer Family 06 ! $4.9 B First Allied Corporation , Tampa Bay Buccaneers Newcastle United ( more information ) Mike Ashley 04 ! $3.3 B Sports Direct International Southampton Gao Jisheng ( 80 % ) Katharina Liebherr ( 20 % ) 07 ! $3.9 B + Lander Sports Development Co Ltd Inheritance Stoke City The Coates Family 08 ! $3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B -- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport , Sunday Sport Gold Group International Straumur Investment Bank
  Fact 0.0: Arsenal's owner E. Stanley Kroenke holds 67.05% of the club.
  Fact 0.1: Arsenal's owner Alisher Usmanov holds 30.04% of the club.
  Fact 0.2: Arsenal's estimated combined owner net worth is $22.4 B.
  Fact 0.3: Arsenal's owners' source of wealth is metal and mining and investment.
  Fact 0.4: Kroenke Sports Enterprises is listed for Arsenal.
  Fact 0.5: Bournemouth's owner Maxim Demin holds 75% of the club.
  Fact 0.6: Bournemouth's owner PEAK6 Investments holds 25% of the club.
  Fact 0.7: Bournemouth's source of wealth is petrochemicals trading.
  Fact 0.8: Seven Developments is listed for Bournemouth.
  Fact 0.9: Brighton & Hove Albion's owner Tony Bloom holds 75.61% of the club.
  Fact 0.10: Brighton & Hove Albion's owner net worth is $1.7 B.
  Fact 0.11: Brighton & Hove Albion's source of wealth is online gambling, real estate, property, land development and investments.
  Fact 0.12: Burnley's owner Mike Garlick holds 49.3% of the club.
  Fact 0.13: Burnley's owner John Banaszkiewicz holds 27.55% of the club.
  Fact 0.14: Burnley's estimated combined owner net worth is $80 M.
  Fact 0.15: Burnley's sources of wealth are Michael Bailey Associates and Freight Investor Services.
  Fact 0.16: Chelsea's owner is Roman Abramovich.
  Fact 0.17: Chelsea's owner net worth is $9.3 B.
  Fact 0.18: Chelsea's source of wealth is oil and industry.
  Fact 0.19: Crystal Palace's owners are Steve Parish, Martin Long, Joshua Harris, David S. Blitzer, Jeremy Hosking and Stephen Browett.
  Fact 0.20: Crystal Palace's owners' net worths are listed as $65 M, $100 M, $3.2 B, $1.3 B, $250 M and $110 M.
  Fact 0.21: Crystal Palace's sources of wealth are listed as Tag Worldwide, Churchill Insurance, Apollo Global Management, Blackstone Group, Marathon Asset Management and Farr Vintners.
  Fact 0.22: Everton's owner Farhad Moshiri holds 49.9% of the club.
  Fact 0.23: Everton's owners include Bill Kenwright CBE and Jon Woods.
  Fact 0.24: Everton's estimated combined owner net worth is $2.3 B.
  Fact 0.25: Everton's sources of wealth are steel and energy, theatre production, Planet Hollywood and Ocean Software.
  Fact 0.26: Huddersfield Town's owner is Dean Hoyle.
  Fact 0.27: Huddersfield Town's owner net worth is $475 M.
  Fact 0.28: Huddersfield Town's source of wealth is Card Factory.
  Fact 0.29: Leicester City's owner is The Srivaddhanaprabha Family.
  Fact 0.30: Leicester City's owner net worth is $4.9 B.
  Fact 0.31: Leicester City's source of wealth is King Power International Group.
  Fact 0.32: Liverpool's owner is John W. Henry.
  Fact 0.33: Liverpool's owner net worth is $2.5 B.
  Fact 0.34: Liverpool's source of wealth is Fenway Sports Group.
  Fact 0.35: Manchester City's owner Mansour bin Zayed Al Nahyan holds 87% of the club.
  Fact 0.36: Manchester City's owner China Media Capital holds 13% of the club.
  Fact 0.37: Manchester City's owner net worth is $44 B.
  Fact 0.38: Manchester City's sources of wealth are oil and industry investment and government.
  Fact 0.39: Manchester United's owner is The Glazer Family.
  Fact 0.40: Manchester United's owner net worth is $4.9 B.
  Fact 0.41: Manchester United's sources of wealth are First Allied Corporation and Tampa Bay Buccaneers.
  Fact 0.42: Newcastle United's owner is Mike Ashley.
  Fact 0.43: Newcastle United's owner net worth is $3.3 B.
  Fact 0.44: Newcastle United's source of wealth is Sports Direct International.
  Fact 0.45: Southampton's owner Gao Jisheng holds 80% of the club.
  Fact 0.46: Southampton's owner Katharina Liebherr holds 20% of the club.
  Fact 0.47: Southampton's owner net worth is $3.9 B.
  Fact 0.48: Southampton's source of wealth is Lander Sports Development Co Ltd and inheritance.
  Fact 0.49: Stoke City's owner is The Coates Family.
  Fact 0.50: Stoke City's owner net worth is $3.1 B.
  Fact 0.51: Stoke City's source of wealth is Bet365.
  Fact 0.52: Swansea City's owners Stephen Kaplan and Jason Levien hold 68% of the club.
  Fact 0.53: Swansea City's Supporters Trust holds 21.1% of the club.
  Fact 0.54: Swansea City's sources of wealth are Memphis Grizzlies, D.C. United and investment trust.
  Fact 0.55: Tottenham Hotspur's owner Joe Lewis holds 70.6% of the 85% holding company.
  Fact 0.56: Tottenham Hotspur's owner Daniel Levy holds 29.4% of the 85% holding company.
  Fact 0.57: Tottenham Hotspur's owner net worth is $5.6 B.
  Fact 0.58: Tottenham Hotspur's sources of wealth are Tavistock Group and ENIC Group.
  Fact 0.59: Watford's owner is Gino Pozzo.
  Fact 0.60: Watford's owner is described as a businessman.
  Fact 0.61: West Bromwich Albion's owner Guochuan Lai holds 87.8% of the club.
  Fact 0.62: West Bromwich Albion's owner net worth is $6.2 B.
  Fact 0.63: West Bromwich Albion's source of wealth is investment.
  Fact 0.64: West Ham United's owner David Sullivan holds 51.1% of the club.
  Fact 0.65: West Ham United's owner David Gold holds 35.1% of the club.
  Fact 0.66: West Ham United's owner Straumur Investment Bank holds 10% of the club.
  Fact 0.67: West Ham United's owners' net worths are listed as $520 M and $325 M.
  Fact 0.68: West Ham United's sources of wealth are Daily Sport, Sunday Sport, Gold Group International and Straumur Investment Bank.
Sentence 1: Club Owner ( s ) Estimated combined net worth ( £ m ) Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % ) Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % ) Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company Middlesbrough Steve Gibson 10 ! $260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn - Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C. Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 ! $155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment
  Fact 1.0: Aston Villa's owner is Dr. Tony Xia.
  Fact 1.1: Aston Villa's owner net worth is $1.2 B.
  Fact 1.2: Aston Villa's source of wealth is Recon Group.
  Fact 1.3: Barnsley's owner is Patrick Cryne.
  Fact 1.4: Barnsley's owner net worth is $40 M.
  Fact 1.5: Barnsley's source of wealth is iSOFT.
  Fact 1.6: Birmingham City's owners are Paul Suen (60.78%), Wang Lei (5.28%) and Carson Yeung (6.51%).
  Fact 1.7: Birmingham City's estimated combined owner net worth is $570 M.
  Fact 1.8: Birmingham City's source of wealth is Birmingham International Holdings.
  Fact 1.9: Bolton Wanderers' owner is Eddie Davies CBE.
  Fact 1.10: Bolton Wanderers' owner net worth is $52 M.
  Fact 1.11: Bolton Wanderers' source of wealth is Strix.
  Fact 1.12: Brentford's owner Matthew Benham holds 96% of the club.
  Fact 1.13: Bristol City's owner is Stephen Lansdown.
  Fact 1.14: Bristol City's owner net worth is $1.9 B.
  Fact 1.15: Bristol City's source of wealth is Hargreaves Lansdown.
  Fact 1.16: Burton Albion's owner is Ben Robinson.
  Fact 1.17: Cardiff City's owner Vincent Tan holds 51% of the club.
  Fact 1.18: Cardiff City's owner net worth is $820 M.
  Fact 1.19: Cardiff City's source of wealth is Berjaya Group.
  Fact 1.20: Derby County's owner is Mel Morris.
  Fact 1.21: Derby County's owner net worth is $520 M.
  Fact 1.22: Fulham's owner is Shahid Khan.
  Fact 1.23: Fulham's owner net worth is $8.9 B.
  Fact 1.24: Fulham's source of wealth is Flex-N-Gate Corp, an automobile parts manufacturer.
  Fact 1.25: Hull City's owner is The Allam Family.
  Fact 1.26: Hull City's owner net worth is $325 M.
  Fact 1.27: Hull City's source of wealth is Allam Marine.
  Fact 1.28: Ipswich Town's owner is Marcus Evans.
  Fact 1.29: Ipswich Town's owner net worth is $825 M.
  Fact 1.30: Ipswich Town's source of wealth is Marcus Evans Group.
  Fact 1.31: Leeds United's owners are Andrea Radrizzani, Eleonora Sports Ltd and Aser Group Company.
  Fact 1.32: Middlesbrough's owner is Steve Gibson.
  Fact 1.33: Middlesbrough's owner net worth is $260 M.
  Fact 1.34: Middlesbrough's source of wealth is Bulkhaul Limited.
  Fact 1.35: Millwall's owners are John Berylson and Richard Smith.
  Fact 1.36: Millwall's owner net worth is $175 M.
  Fact 1.37: Millwall's source of wealth is Chestnut Hill Ventures.
  Fact 1.38: Norwich City's owners Delia Smith and Michael Wynn-Jones hold 53% of the club.
  Fact 1.39: Norwich City's owner Michael Foulger holds 15% of the club.
  Fact 1.40: Norwich City's owner net worth is $30 M.
  Fact 1.41: Norwich City's sources of wealth are food industry, publishing and poultry.
  Fact 1.42: Nottingham Forest's owner is Evangelos Marinakis.
  Fact 1.43: Nottingham Forest's owner net worth is $650 M.
  Fact 1.44: Nottingham Forest's owner is also owner of Olympiacos F.C.
  Fact 1.45: Preston North End's owner Trevor Hemmings CVO holds 51.4% of the club.
  Fact 1.46: Preston North End's owner net worth is $720 M.
  Fact 1.47: Preston North End's sources of wealth are Blackpool Tower, horse racing, investment, Northern Trust and Sportech.
  Fact 1.48: Queens Park Rangers' owner Tony Fernandes holds 66% of the club.
  Fact 1.49: Queens Park Rangers' owner Lakshmi Mittal holds 33% of the club.
  Fact 1.50: Queens Park Rangers' owners' net worths are listed as $345 M and $16.4 B.
  Fact 1.51: Queens Park Rangers' sources of wealth are Air Asia, Tune Group and ArcelorMittal.
  Fact 1.52: Reading's owners are Narin Niruttinanon (50%), Sasima Srivikorn (25%) and Sumrith Thanakarnjanasuth (25%).
  Fact 1.53: Sheffield United's owner Abdullah bin Musa'ed holds 50% of the club.
  Fact 1.54: Sheffield United's owner net worth is $155 M.
  Fact 1.55: Sheffield United's sources of wealth are property and oil.
  Fact 1.56: Sheffield Wednesday's owner is Dejphon Chansiri.
  Fact 1.57: Sheffield Wednesday's source of wealth is Thai Union Group.
  Fact 1.58: Sunderland's owner is Ellis Short.
  Fact 1.59: Sunderland's owner net worth is $2.9 B.
  Fact 1.60: Sunderland's sources of wealth are Lone Star Funds and Skibo Castle.
  Fact 1.61: Wolverhampton Wanderers' owners are Fosun International (parent company) and Guo Guangchang (co-owner of Fosun).
  Fact 1.62: Wolverhampton Wanderers' owner net worth is $6.2 B.
  Fact 1.63: Wolverhampton Wanderers' source of wealth is investment.

Retrieved items:
<item 1>
List of owners of English football clubs
Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % )
Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % )
Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby
County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam
Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company
Middlesbrough Steve Gibson 10 !
</item 1>

<item 2>
List of owners of English football clubs
$22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven
Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike
Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman
Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M
$3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more
information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 !
</item 2>

<item 3>
List of owners of English football clubs
A.F.C. Sunderland . Retrieved 9 May 2014 . Jump up ^ `` How the English Premier League Clubs ' Owners Got Rich '' . Archived from the original on 13 January
2015 . Retrieved 11 December 2016 . Jump up ^ `` # 226 Guo Guangchang '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Bury FC Ownership '' . Bury F.C. 25
June 2012 . Retrieved 12 February 2013 . Jump up ^ `` Directors '' . Charlton Athletic F.C. 30 July 2013 . Retrieved 12 May 2014 . Jump up ^ `` Who 's Who at
Doncaster Rovers '' . Doncaster Rovers F.C . Retrieved 12 February 2013 . Jump up ^ `` Terry Bramall Awarded CBE '' . Doncaster Rovers F.C. 29 December 2012 .
Retrieved 13 February 2013 . Jump up ^ `` Ownership '' . Fleetwood Town F.C .
</item 3>

<item 4>
List of owners of English football clubs
$250 M DW Sports Fitness , JJB Sports League Two ( edit ) Club Owner ( s ) Estimated combined net worth Source of wealth Accrington Stanley The Styring Family (
12 % ) Accrington Stanley Supporters Fund ( 12 % ) Peter Marsden ( 10 % ) Investment trust Barnet Blackpool Owen Oyston ( 80 % ) Valeri Belokon ( 25 % ) 07 !
$105 M $195 M Publishing , Media & Estate Agency Finance , Media , Food Industry & Mining Cambridge United Grosvenor Fund Management Grosvenor Group Owned by
the Duke of Westminster and family Property Carlisle United ( more information ) A. Jenkins ( 37 % of 93 % holding company ) J. Nixon ( 18 % ) S. Pattison ( 18
% ) The United Trust ( 25.4 % ) Pioneer Foods Businessman ( retired ) Carlisle Glass Investment Trust Cheltenham Town Paul Baker Chesterfield Dave Allen
Colchester United Robbie Cowling 09 !
</item 4>

<item 5>
List of owners of English football clubs
$3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United
Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B
-- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment
West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport ,
Sunday Sport Gold Group International Straumur Investment Bank Football League Championship ( edit ) Club Owner ( s ) Estimated combined net worth ( £ m )
Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B !
</item 5>

<item 6>
List of owners of English football clubs
Jump up ^ `` Shahid Khan '' . Forbes . Forbes.com LLC. 19 May 2017 . Retrieved 19 May 2017 . Jump up ^ `` Who 's Who At The KC '' . Hull City F.C. 25 June 2012
. Retrieved 9 May 2014 . Jump up ^ `` Sunday Times Rich List 2017 reveals Yorkshire 's 20 wealthiest people '' . Hull Daily Mail . Archived from the original on
2 May 2017 . Retrieved 19 May 2017 . Jump up ^ www.calcioefinanza.it Leeds http://www.calcioefinanza.it/2017/01/04/leeds-cellino-radrizzani/ Leeds Check url =
value ( help ) . Retrieved 2017 - 01 - 10 . Missing or empty title = ( help ) Jump up ^ leeds united
https://www.leedsunited.com/news/club-news/21921/club-statement . Retrieved 2017 - 05 - 23 .
</item 6>

<item 7>
List of owners of English football clubs
^ Jump up to : `` The Football Rich List 2011 / 12 : The Full 100 '' . Four Four Two . Retrieved 13 February 2013 . Jump up ^ `` Board of Directors / CEO '' .
Burnley F.C . Retrieved 9 May 2014 . Jump up ^ `` The Rich List 2014 '' . Hertfordshire Life . 31 March 2014 . Retrieved 12 May 2014 . Jump up ^ `` Club
Personnel '' . Chelsea F.C . Retrieved 9 August 2015 . Jump up ^ `` # 139 Roman Abramovich '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` # 745 Joshua
Harris '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Senior changes at Tag Worldwide '' . Williams Lea. 24 October 2012 . Retrieved 16 February 2013 . Jump
up ^ `` # 896 Farhad Moshiri '' . Forbes . Retrieved 19 May 2017 .
</item 7>

<item 8>
List of owners of English football clubs
$155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton
Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment League One ( edit ) Club Owner ( s ) Estimated combined
net worth Source of wealth AFC Wimbledon Wimbledon Football Club Supporters ' Society ( The Dons Trust ) Supporters Trust Blackburn Rovers Balaji & Venkatesh
Rao ( 99.9 % ) 04 ! $650 M Venkateshwara Hatcheries Group Bradford City Bristol Rovers Dwane Sports ( 92 % ) Banking and tourism Bury Save Our Shakers Trust (
63.8 % ) The Bury F.C. Supporters Society Ltd ( Forever Bury ) ( 11 % ) Investment trusts Charlton Athletic Roland Duchatelet ( 95 % ) 24 ! Staprix NV Doncaster
Rovers John Ryan Dick Watson ( 26 % ) Terry Bramall CBE 07 !
</item 8>

<item 9>
List of owners of English football clubs
List of owners of English football clubs - wikipedia List of owners of English football clubs Jump to : navigation , search This is a list of the current owners
of English football clubs , as well as ( in some cases ) their estimated net worth and source of wealth . All figures are listed in millions rather than
billions . For example , £ 1,000 m would be one billion ( 1,000,000,000 ) pounds . Only shareholders with a significant interest ( above 10 % ) are listed .
Contents ( hide ) 1 Premier League 2 Football League Championship 3 League One 4 League Two 5 References Premier League ( edit ) This list is incomplete ; you
can help by expanding it . Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher
Usmanov ( 30.04 % ) 15 !
</item 9>

<item 10>
List of owners of English football clubs
$260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn -
Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C.
Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more
information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin
Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 !
</item 10>
=== END ITEM id=d97961b24c7512a2 ===

=== ITEM id=35933af236d210c6 ===
Gold sentences and facts:
Sentence 0: The 1836 article and other sources make it clear that this verse was not used soon after 1745 , and certainly before the song became accepted as the British national anthem in the 1780s and 1790s . It was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926 , although erroneously referencing the `` fourth verse '' to the Gentleman 's Magazine article of 1745 .
  Fact 0.0: The 1836 article and other sources show that this verse was not used soon after 1745.
  Fact 0.1: The 1836 article and other sources show that this verse was certainly not used before the song became accepted as the British national anthem in the 1780s and 1790s.
Sentence 1: The French Marquise de Créquy wrote in her Souvenirs that the words `` Grand Dieu , sauvez le Roi ! '' were indeed composed by Jean - Baptiste Lully in gratitude for the survival by Louis XIV of an anal fistula operation . ( The surgical knife purpose - made for the occasion is on display in the Musée d'histoire de la médecine . ) Allegedly , Lully had set words by Marie de Brinon to music , and Créquy claimed that the tune was later plagiarised by Handel , and so adopted by the British as `` un de leurs air nationaux '' . ( The entire story might have been intended as a joke . ) Translated into Latin under the title Domine , Salvum Fac Regem , it became the French anthem until 1792 . After the Battle of Culloden , the Hanover dynasty supposedly adopted this melody as the British anthem .
  Fact 1.0: The verse was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926.
  Fact 1.1: The Oxford Book of Eighteenth Century Verse of 1926 erroneously referenced the 'fourth verse' to the Gentleman's Magazine article of 1745.

Retrieved items:
<item 1>
God Save the Queen
Republic . Retrieved 11 November 2017 . External links ( edit ) Wikisource has original text related to this article : God Save the Queen Wikimedia Commons has
media related to God Save the Queen . National Anthem at the Royal Family website Streaming audio , lyrics and information about God Save the Queen Department
of Canadian Heritage -- Royal anthem page God Save Great George our King : -- article discussing different versions of the lyrics Free sheet music of God Save
the Queen from Cantorion.org A Point of View : Is it time for a new British national anthem ? BBC NEWS . Published 15 January 2016 . Retrieved 17 July 2017 .
National anthems of Europe Sovereign states Albania Andorra Armenia Austria Azerbaijan Belarus Belgium Bosnia and Herzegovina Bulgaria Croatia Cyprus Czech
Republic Denmark Estonia Finland France Georgia Germany Greece Hungary Iceland Ireland Italy Kazakhstan Latvia Liechtenstein Lithuania Luxembourg Macedonia
Malta Moldova Monaco Montenegro Netherlands Norway Poland Portugal Romania Russia San Marino Serbia Slovakia Slovenia Spain Sweden Switzerland Turkey Ukraine
United Kingdom Vatican City Territories , dependencies , other areas United Kingdom Constituent countries England Northern Ireland Scotland Wales British
Overseas Territories Gibraltar Crown dependencies Guernsey Isle of Man Jersey Counties Cornwall Lincolnshire Yorkshire Russia Adygea
</item 1>

<item 2>
God Save the Queen
In some countries it remains as one of the official national anthems , such as in New Zealand , or as an official royal anthem , as is the case in Australia ,
Canada , Jamaica , and Tuvalu , to be played during formal ceremonies involving national royalty or vice-royalty . Australia ( edit ) Further information :
Australian royal symbols § Verbal and musical symbols In Australia , the song has standing through a Royal Proclamation issued by Governor - General Sir Ninian
Stephen on 19 April 1984 . It declared `` God Save the Queen '' to be the Royal Anthem and that it is to be played when the Australian monarch or a member of
the Royal Family is present , though not exclusively in such circumstances . The same proclamation made `` Advance Australia Fair '' the national anthem and the
basis for the `` Vice-Regal Salute '' ( the first four and last two bars of the anthem ) . Prior to 1984 , `` God Save the Queen '' was the national anthem of
Australia . Canada ( edit ) Further information : Canadian royal symbols § Verbal and musical symbols , and Anthems and nationalistic songs of Canada By
convention , `` God Save the Queen '' is the Royal Anthem of Canada .
</item 2>

<item 3>
God Save the Queen
It is the national anthem of the United Kingdom and one of two national anthems used by New Zealand since 1977 , as well as for several of the UK 's territories
that have their own additional local anthem . It is also the royal anthem -- played specifically in the presence of the monarch -- of all the aforementioned
countries , as well as Australia ( since 1984 ) , Canada ( since 1980 ) , Barbados and Tuvalu . In countries not previously part of the British Empire , the
tune of `` God Save the Queen '' has provided the basis for various patriotic songs , though still generally connected with royal ceremony . In the United
States , the melody is used for the patriotic song `` My Country , ' Tis of Thee '' . The melody is also used for the national anthem of Liechtenstein , `` Oben
am jungen Rhein '' . Beyond its first verse , which is consistent , `` God Save the Queen / King '' has many historic and extant versions . Since its first
publication , different verses have been added and taken away and , even today , different publications include various selections of verses in various orders .
In general , only one verse is sung .
</item 3>

<item 4>
God Save the Queen
In April 2007 there was an early day motion , number 1319 , to the British Parliament to propose that there should be a separate England anthem : `` That this
House ... believes that all English sporting associations should adopt an appropriate song that English sportsmen and women , and the English public , would
favour when competing as England '' . An amendment ( EDM 1319A3 ) was proposed by Evan Harris that the song `` should have a bit more oomph than God Save The
Queen and should also not involve God . '' For more information see also : National anthem of England National anthem of Scotland Hen Wlad Fy Nhadau Londonderry
Air Use in other Commonwealth countries ( edit ) `` God Save the King / Queen '' was exported around the world via the expansion of the British Empire , serving
as each country 's national anthem . Throughout the Empire 's evolution into the Commonwealth of Nations , the song declined in use in most states which became
independent .
</item 4>

<item 5>
God Save the Queen
The 1836 article and other sources make it clear that this verse was not used soon after 1745 , and certainly before the song became accepted as the British
national anthem in the 1780s and 1790s . It was included as an integral part of the song in the Oxford Book of Eighteenth Century Verse of 1926 , although
erroneously referencing the `` fourth verse '' to the Gentleman 's Magazine article of 1745 . On the opposing side , Jacobite beliefs were demonstrated in an
alternative verse used during the same period : God bless the prince , I pray , God bless the prince , I pray , Charlie I mean ; That Scotland we may see Freed
from vile Presbyt'ry , Both George and his Feckie , Ever so , Amen . In May 1800 , following an attempt to assassinate King George III at London 's Drury Lane
theatre , playwright Richard Sheridan immediately composed an additional verse , which was sung from the stage the same night : From every latent foe From the
assassins blow God save the King O'er him Thine arm extend For Britain 's sake defend Our father , king , and friend God save the King !
</item 5>

<item 6>
God Save the Queen
National or royal anthem of Antigua and Barbuda ( royal ) Australia ( royal ) and its dependency of Norfolk Island ( national and royal ) The Bahamas ( royal )
Barbados ( royal ) Belize ( royal ) Canada ( royal ) Grenada ( royal ) Jamaica ( royal ) New Zealand ( national and royal ) and some of its dependencies ( Cook
Islands and Niue ( royal ) and Tokelau ( national and royal ) ) Saint Kitts and Nevis ( royal ) Mauritius ( royal ) 1968 - 1992 Solomon Islands ( royal ) Tuvalu
( royal ) United Kingdom ( national and royal ) , its Crown Dependencies and its overseas territories Also known as `` God Save the King '' ( when the monarch
is male ) Music Unknown composer Adopted September 1745 Audio sample `` God Save the Queen '' file help `` God Save the Queen '' ( alternatively `` God Save the
King '' , depending on the gender of the reigning monarch ) is the national or royal anthem in a number of Commonwealth realms , their territories , and the
British Crown dependencies . The author of the tune is unknown , and it may originate in plainchant ; but an attribution to the composer John Bull is sometimes
made .
</item 6>

<item 7>
God Save the Queen
3.2 Official peace version 2.1. 4 Historic alternative verses 2.2 Performance in the UK 2.3 Other British anthems 3 Use in other Commonwealth countries 3.1
Australia 3.2 Canada 3.2. 1 Lyrics in Canada 3.3 New Zealand 3.4 Rhodesia 4 Use elsewhere 5 Musical adaptations 5.1 Classical composers 5.2 Rock adaptations 5.3
In popular culture 5.4 Computer music 6 Reception 6.1 Calls for a new national anthem / anthems 7 Notes 8 External links History ( edit ) In The Oxford
Companion to Music , Percy Scholes points out the similarities to an early plainsong melody , although the rhythm is very distinctly that of a galliard , and he
gives examples of several such dance tunes that bear a striking resemblance to `` God Save the King / Queen '' . Scholes quotes a keyboard piece by John Bull (
1619 ) which has some similarities to the modern tune , depending on the placing of accidentals which at that time were unwritten in certain cases and left to
the discretion of the player ( see musica ficta ) .
</item 7>

<item 8>
God Save the Queen
The first music proper , a performance of the National Anthem was programmed by Christopher Strachey on the Mark II Manchester Electronic Computer at same venue
, in 1951 . Later that year , short extracts of three pieces , the first being the National Anthem , were recorded there by a BBC outside broadcasting unit :
the other pieces being `` Ba Ba Black Sheep , and `` In the Mood '' . Researchers at the University of Canterbury , Christchurch restored the acetate master
disc in 2016 and the results may be heard on Soundcloud . Reception ( edit ) The philosopher and reformer Jeremy Bentham praised `` God Save the King '' in 1796
: `` the melody recommending itself by beauty to the most polished ears , and by its simplicity to the rudest ear . A song of this complexion , implanted by the
habit of half a century in the mass of popular sentiment , can not be refused a place in the inventory of the national blessings . '' Ludwig van Beethoven wrote
`` I have to show the English a little of what a blessing ' God Save the King ' is '' . Alex Marshall , the British author of Republic or Death !
</item 8>

<item 9>
God Save the Queen
Send him victorious Happy and glorious Long to reign over us God save the king ! Use in the United Kingdom ( edit ) See also : National anthem of England The
phrase `` God Save the King '' in use as a rallying cry to the support of the monarch and the UK 's forces `` God Save the Queen '' is the national anthem of
the United Kingdom of Great Britain and Northern Ireland . Like many aspects of British constitutional life , its official status derives from custom and use ,
not from Royal Proclamation or Act of Parliament . In general only one or two verses are sung , but on rare occasions three . The variation in the UK of the
lyrics to `` God Save the Queen '' is the oldest amongst those currently used , and forms the basis on which all other versions used throughout the Commonwealth
are formed ; though , again , the words have varied throughout these years . England has no official national anthem of its own ; `` God Save the Queen '' is
treated as the English national anthem when England is represented at sporting events ( though there are some exceptions to this rule , such as cricket where
Jerusalem is used ) .
</item 9>

<item 10>
God Save the Queen
Since 1953 , the anthem is sometimes preceded by a fanfare composed by Gordon Jacob for the coronation of Queen Elizabeth II . Alternative British versions (
edit ) There have been several attempts to improve the song by rewriting the words . In the nineteenth century there was some lively debate about the national
anthem and , even then , verse two was considered to be slightly offensive . Notably , the question arose over the phrase `` scatter her enemies . '' Some
thought it placed better emphasis on the respective power of Parliament and the Crown to change `` her '' to `` our '' ; others pointed out that the theology
was somewhat dubious and substituted `` thine '' instead . Sydney G.R. Coles wrote a completely new version , as did Canon F.K. Harford . In 1836 , William
Edward Hickson wrote four alternative verses . The first , third , and fourth of these verses are appended to the National Anthem in the English Hymnal ( which
only includes verses one and three of the original lyrics ) .
</item 10>
=== END ITEM id=35933af236d210c6 ===

=== ITEM id=afd9387e4574c506 ===
Gold sentences and facts:
Sentence 0: The Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the `` Watchtower '' and the `` Swamp '' .
  Fact 0.0: The Curse of Oak Island is a series.
  Fact 0.1: The Curse of Oak Island follows brothers Marty and Rick Lagina.
  Fact 0.2: Marty and Rick Lagina are originally from Kingsford, Michigan.
  Fact 0.3: The Curse of Oak Island follows the Laginas' efforts to find the speculated treasure or historical artifacts believed to be on Oak Island.
  Fact 0.4: The series discusses the history of Oak Island.
  Fact 0.5: The series discusses recent discoveries on Oak Island.
  Fact 0.6: The series discusses theories about Oak Island.
  Fact 0.7: The series discusses prior attempts to investigate the site on Oak Island.
  Fact 0.8: Areas of interest in the series include the "Money Pit".
  Fact 0.9: Areas of interest in the series include Borehole 10-x.
  Fact 0.10: Areas of interest in the series include Smith's Cove.
  Fact 0.11: Areas of interest in the series include "Nolan's Cross".
  Fact 0.12: Areas of interest in the series include the "Hatch".
  Fact 0.13: Areas of interest in the series include the "Watchtower".
  Fact 0.14: Areas of interest in the series include the "Swamp".

Retrieved items:
<item 1>
The Curse of Oak Island
Contents ( hide ) 1 Overview 1.1 Background 1.2 Theories 2 Series overview 3 Production 4 Episodes 4.1 Season 1 ( 2014 ) 4.2 Season 2 ( 2014 -- 2015 ) 4.3
Season 3 ( 2015 -- 2016 ) 4.4 Season 4 ( 2016 -- 2017 ) 4.5 Season 5 ( 2017 -- 2018 ) 5 References 6 Further reading 7 External links Overview ( edit ) The
Curse of Oak Island follows brothers Marty and Rick Lagina , originally from Kingsford , Michigan , through their efforts to find the speculated treasure or
historical artifacts believed to be on Oak Island . The series discusses the history of the island , recent discoveries , theories , and prior attempts to
investigate the site . Areas of interest include the `` Money Pit '' , Borehole 10 - x , Smith 's Cove , `` Nolan 's Cross '' , the `` Hatch '' , the ``
Watchtower '' and the `` Swamp '' .
</item 1>

<item 2>
The Curse of Oak Island
On August 2 , 2016 , it was announced that Nova Scotia Business Inc. approved $1,271,546 in film funding for the production of the 4th season . In October 2017
, a fifth season was announced , which began airing on November 7 , 2017 . Episodes ( edit ) Season 1 ( 2014 ) ( edit ) No . overall No. in season Title
Original air date Viewers ( millions ) `` What Lies Below '' January 5 , 2014 ( 2014 - 01 - 05 ) 2.53 Brothers Rick and Marty Lagina now own most of Oak Island
. Determined to resume the search for treasure on the island , they explore an abandoned shaft called Borehole 10 - X and use water pressure to pump material to
the surface . They find a tiny piece of metal .
</item 2>

<item 3>
The Curse of Oak Island
Lee Lamb and Richard Restall visit the island . Richard Restall is identified conclusively as the owner of the toy gun . 49 5 `` Bone Dry '' December 5 , 2017 (
2017 - 12 - 05 ) 3.16 An iron spike is found on the beach . Some small pottery fragments are brought up from one of the boreholes . Two human bone fragments
were recovered and sent in for testing . 50 6 `` Remains of the Day '' December 12 , 2017 ( 2017 - 12 - 12 ) 3.01 A piece of leather is found . Rick Lagina has
headaches and a rash . He is later diagnosed as having lyme disease . The team intends to have the previously discovered human bone fragments carbon dated . 51
7 `` The Lot Thickens '' December 19 , 2017 ( 2017 - 12 - 19 ) 3.13 A theory is presented that Sir Francis Drake is buried on Oak Island in a vault with gold
treasure . It is suggested that the lack of any records documenting Drake travelling to Canada is proof that he must have travelled to Canada .
</item 3>

<item 4>
The Curse of Oak Island
overall No. in season Title Original air date Viewers ( millions ) 29 `` Going for Broke '' November 15 , 2016 ( 2016 - 11 - 15 ) 2.62 The team agrees to turn
focus over to the Money Pit and review what they have collected . Researcher Zena Halpern presents a handmade copy of what she claims is a French map from 1347
of Oak Island . The map shows locations for a `` hatch '' , a `` valve '' and an `` anchor '' on the Island . A theory is presented based on a purported
decoding of a cipher that Joab at the time of King David transported the Ark of the Covenant to North Africa from which it was later transported onward by the
Knights Templar from the then - Portuguese town of Safi in Morocco to Oak Island . 30 `` Always Forward '' November 22 , 2016 ( 2016 - 11 - 22 ) 2.66 While
preparing to excavate the Money Pit , the team investigates an 18th century well on mainland Nova Scotia .
</item 4>

<item 5>
The Curse of Oak Island
Background ( edit ) The Lagina brothers became fascinated with the island after reading the January 1965 issue of Reader 's Digest magazine that features an
article on the Restall family 's work to investigate the so - called `` Money Pit '' . Marty and his brother Rick obtained a controlling interest in Oak Island
Tours , which reportedly owns most of the island . The brothers were later approached by Prometheus Entertainment to do a reality show . Rick and Marty have
engaged the assistance of father - and - son Dan and Dave Blankenship , permanent residents of the island who have likewise been searching for treasure since
the 1960s . Theories ( edit ) The series explores various Oak Island theories and conjectures through conversations with independent researchers .
</item 5>

<item 6>
The Curse of Oak Island
26 11 `` Sword Play '' January 19 , 2016 ( 2016 - 01 - 19 ) 2.82 The team has the Roman sword investigated and find that it is a modern reproduction . There is
an unproductive search for treasure north of the Money Pit . 27 12 `` Voices from Below '' January 26 , 2016 ( 2016 - 01 - 26 ) 3.17 The team dives to the
bottom of borehole 10X . In a separate investigation , a cavity is found far underground near the Money Pit . 28 13 `` Secrets and Revelations '' February 2 ,
2016 ( 2016 - 02 - 02 ) 3.41 The diving investigation of borehole 10X concludes with nothing found . Relatives of the original Oak Island treasure hunters
present a gold cross which is said to have come from the treasure . Season 4 ( 2016 -- 2017 ) ( edit ) No . overall No.
</item 6>

<item 7>
The Curse of Oak Island
Reader 's Digest -- via readersdigest.ca . External links ( edit ) Official website Official Facebook Page The Curse of Oak Island on IMDb History original
programming Current Alone American Pickers American Restoration Ancient Aliens Counting Cars The Curse of Oak Island Forged in Fire Hunting Hitler Ice Road
Truckers Knightfall Mountain Men Ozzy & Jack 's World Detour Pawn Stars Six Swamp People Vikings Former 20th Century with Mike Wallace Appalachian Outlaws
Alaska Off - Road Warriors America Unearthed American Eats Ancient Discoveries Ancients Behaving Badly Around the World in 80 Ways Ax Men Barbarians Battle 360
° Battlefield Detectives Battles BC Big Easy Motors Big Rig Bounty Hunters Big Shrimpin ' Billion Dollar Wreck Boneyard Brad Meltzer 's Decoded Breaking Vegas
Cajun Pawn Stars Cities of the Underworld Civil War Combat Clash of the Gods The Conquerors Conquest The Crusades : Crescent and the Cross Decisive Battles
Declassified Decoding the Past Deep Sea Detectives Digging for the Truth Dogfights Down East Dickering Engineering an Empire Evolve Expedition Africa Extreme
Trains Food Tech Full Metal Jousting Gangland Gangland Undercover Great Minds with Dan Harmon Greatest Tank Battles God ,
</item 7>

<item 8>
The Curse of Oak Island
61 17 `` A Family Album '' February 27 , 2018 ( 2018 - 02 - 27 ) 1.98 The episode includes accounts of activities on the Oak Island by the McGinnis family ,
descendants of Samuel Ball , William Chappell 's family , the Restall Family and the family of Robert Dunfield . The Blankenship / Nolan disputes on Oak Island
are discussed . The episode ends with a short history of the Lagina brothers ' interest in Oak Island . Note : This episode is a special . 62 18 `` Amazing
Discoveries '' March 6 , 2018 ( 2018 - 03 - 06 ) 2.89 The gem in the Jewelry fragment found on the island is a Rhodolite Garnet stone of small value . The
drilling obstruction in the `` money pit '' borehole is determined by divers to be a granite boulder . References ( edit ) ^ Jump up to : `` Shows A-Z - curse
of oak island , the on history '' . The Futon Critic . Retrieved December 6 , 2017 .
</item 8>

<item 9>
The Curse of Oak Island
The Curse of Oak Island - wikipedia The Curse of Oak Island Jump to : navigation , search The Curse of Oak Island Genre Reality Narrated by Robert Clotworthy
Country of origin Canada Original language ( s ) English No. of seasons 5 No. of episodes 62 Production Executive producer ( s ) Kevin Burns Camera setup
Multiple Running time 43 minutes Production company ( s ) Prometheus Entertainment Distributor A&E Networks Release Original network History Canada Original
release January 5 , 2014 ( 2014 - 01 - 05 ) -- present External links Website Production website The Curse of Oak Island is an active reality television series
that first premiered in Canada on the History network on January 5 , 2014 . The show features what is known as the Oak Island mystery , showing efforts to
search for historical artifacts and treasure .
</item 9>

<item 10>
The Curse of Oak Island
A visitor shows them alleged evidence that the treasures from King Solomon 's Temple could be buried on Oak Island . 8 `` The Eight - Pointed Star '' November
18 , 2014 ( 2014 - 11 - 18 ) 2.23 A treasure hunter presents a theory that Phoenicians visited the island 2500 years ago and left behind the ark of the covenant
. Rick and Marty Lagina unsuccessfully look for evidence to prove it while also trying to locate the original Money Pit . 9 `` The Breakthrough '' November 25 ,
2014 ( 2014 - 11 - 25 ) 2.15 A tree stump in the swamp is examined . Core samples extracted by drilling are discovered to contain rocks and wood . 10 5 `` The
90 - Foot Stone '' December 2 , 2014 ( 2014 - 12 - 02 ) 2.46 A visitor comes to the island suggesting that he has discovered a secret method of getting to the
treasure . He claims that Francis Bacon has left behind instructions to the effect that the `` flood tunnels '' can be disabled by pouring corn into the money
pit .
</item 10>
=== END ITEM id=afd9387e4574c506 ===

=== ITEM id=811de12e0db83626 ===
Gold sentences and facts:
Sentence 0: Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name ( Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741 7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743 -- First Lord of the Treasury Whig Carteret 7002502000000000000 ♠ 1 year and 137 days ​ The Right Honourable Henry Pelham FRS MP for Sussex ( 1694 -- 1754 ) 27 August 1743 6 March 1754 1747 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig Broad Bottom I Broad Bottom II 7003384500000000000 ♠ 10 years and 192 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 16 March 1754 16 November 1756 1754 First Lord of the Treasury Leader of the House of Lords Whig Newcastle I 7002977000000000000 ♠ 2 years and 246 days His Grace William Cavendish 4th Duke of Devonshire KG PC ( 1720 -- 1764 ) 16 November 1756 25 June 1757 -- First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Devonshire ​ 1757 Caretaker 7002222000000000000 ♠ 222 days His Grace Thomas Pelham - Holles 1st Duke of Newcastle KG PC FRS ( 1693 -- 1768 ) 2 July 1757 26 May 1762 1761 First Lord of the Treasury Leader of the House of Lords Whig Pitt -- Newcastle George III ( 1760 -- 1820 ) 7003179000000000000 ♠ 4 years and 329 days The Right Honourable John Stuart 3rd Earl of Bute KG PC ( 1713 -- 1792 ) 26 May 1762 8 April 1763 -- First Lord of the Treasury Leader of the House of Lords Tory Bute ( Tory -- Whig ) 7002318000000000000 ♠ 318 days The Right Honourable George Grenville MP for Buckingham ( 1712 -- 1770 ) 16 April 1763 13 July 1765 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Whig ( Grenvillite ) Grenville ( Whig -- Tory ) 7002820000000000000 ♠ 2 years and 89 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 13 July 1765 30 July 1766 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham I 7002383000000000000 ♠ 1 year and 18 days The Right Honourable William Pitt 1st Earl of Chatham PC FRS MP for Bath ( 1708 -- 1778 ) 30 July 1766 14 October 1768 1768 Lord Keeper of the Privy Seal Whig ( Chathamite ) Chatham ( Whig -- Tory ) 7002808000000000000 ♠ 2 years and 77 days His Grace Augustus FitzRoy 3rd Duke of Grafton KG PC ( 1735 -- 1811 ) 14 October 1768 28 January 1770 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Grafton 7002472000000000000 ♠ 1 year and 107 days The Right Honourable Frederick North Lord North KG MP for Banbury ( 1732 -- 1792 ) 28 January 1770 22 March 1782 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Northite ) North 1774 1780 7003443700000000000 ♠ 12 years and 54 days The Most Honourable Charles Watson - Wentworth 2nd Marquess of Rockingham KG PC FRS ( 1730 -- 1782 ) 27 March 1782 1 July 1782 -- First Lord of the Treasury Leader of the House of Lords Whig ( Rockinghamite ) Rockingham II 7001970000000000000 ♠ 97 days The Right Honourable William Petty 2nd Earl of Shelburne KG PC ( 1737 -- 1805 ) 4 July 1782 2 April 1783 -- First Lord of the Treasury Leader of the House of Lords Whig ( Chathamite ) Shelburne ( Whig -- Tory ) 7002273000000000000 ♠ 273 days His Grace William Cavendish - Bentinck 3rd Duke of Portland PC FRS ( 1738 -- 1809 ) 2 April 1783 19 December 1783 -- First Lord of the Treasury Leader of the House of Lords Whig Fox -- North 7002262000000000000 ♠ 262 days The Right Honourable William Pitt the Younger MP for Appleby -- Cambridge University ( 1759 -- 1806 ) 19 December 1783 14 March 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt I 1784 1790 1796 7003629500000000000 ♠ 17 years and 86 days The Right Honourable Henry Addington MP for Devizes ( 1757 -- 1844 ) 17 March 1801 10 May 1804 1801 First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Addington 1802 7003115100000000000 ♠ 3 years and 55 days The Right Honourable William Pitt the Younger MP for Cambridge University ( 1759 -- 1806 ) 10 May 1804 23 January 1806 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Pittite ) Pitt II 7002624000000000000 ♠ 1 year and 259 days The Right Honourable William Grenville 1st Baron Grenville PC PC ( Ire ) ( 1759 -- 1834 ) 11 February 1806 31 March 1807 First Lord of the Treasury Leader of the House of Lords Whig All the Talents ( Whig -- Tory ) 1806 7002414000000000000 ♠ 1 year and 49 days His Grace William Cavendish - Bentinck 3rd Duke of Portland KG PC FRS ( 1738 -- 1809 ) 31 March 1807 4 October 1809 First Lord of the Treasury Tory ( Pittite ) Portland II 1807 7002919000000000000 ♠ 2 years and 188 days The Right Honourable Spencer Perceval KC MP for Northampton ( 1762 -- 1812 ) 4 October 1809 11 May 1812 -- First Lord of the Treasury Chancellor of the Exchequer Chancellor of the Duchy of Lancaster Leader of the House of Commons Tory ( Pittite ) Perceval 7002951000000000000 ♠ 2 years and 221 days The Right Honourable Robert Jenkinson 2nd Earl of Liverpool KG PC FRS ( 1770 -- 1828 ) 8 June 1812 9 April 1827 1812 First Lord of the Treasury Leader of the House of Lords Tory ( Pittite ) Liverpool 1818 1820 George IV ( 1820 -- 1830 ) 1826 7003541900000000000 ♠ 14 years and 306 days The Right Honourable George Canning FRS MP for Seaford ( 1770 -- 1827 ) 10 April 1827 8 August 1827 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Tory ( Canningite ) Canning ( Canningite -- Whig ) 7002121000000000000 ♠ 121 days The Right Honourable Frederick John Robinson 1st Viscount Goderich PC ( 1782 -- 1859 ) 31 August 1827 21 January 1828 -- First Lord of the Treasury Leader of the House of Lords Tory ( Canningite ) Goderich 7002144000000000000 ♠ 144 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 22 January 1828 16 November 1830 First Lord of the Treasury Leader of the House of Lords Tory Wellington -- Peel 1830 William IV ( 1830 -- 1837 ) 7003103000000000000 ♠ 2 years and 299 days The Right Honourable Charles Grey 2nd Earl Grey KG PC ( 1764 -- 1845 ) 22 November 1830 9 July 1834 First Lord of the Treasury Leader of the House of Lords Whig Grey 1831 1832 7003132600000000000 ♠ 3 years and 230 days The Right Honourable William Lamb 2nd Viscount Melbourne PC ( 1779 -- 1848 ) 16 July 1834 14 November 1834 -- First Lord of the Treasury Leader of the House of Lords Whig Melbourne I 7002122000000000000 ♠ 122 days Field Marshal His Grace Arthur Wellesley 1st Duke of Wellington KG GCB GCH PC ( 1769 -- 1852 ) 14 November 1834 10 December 1834 -- First Lord of the Treasury Sec. of State for the Home Department Sec. of State for Foreign Affairs Sec. of State for War and the Colonies Leader of the House of Lords Tory Wellington Caretaker 7001270000000000000 ♠ 27 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 10 December 1834 8 April 1835 -- First Lord of the Treasury Chancellor of the Exchequer Leader of the House of Commons Conservative Peel I 7002120000000000000 ♠ 120 days The Right Honourable William Lamb 2nd Viscount Melbourne PC FRS ( 1779 -- 1848 ) 18 April 1835 30 August 1841 1835 First Lord of the Treasury Leader of the House of Lords Whig Melbourne II 1837 Victoria ( 1837 -- 1901 ) 7003232700000000000 ♠ 6 years and 135 days The Right Honourable Sir Robert Peel Bt FRS MP for Tamworth ( 1788 -- 1850 ) 30 August 1841 29 June 1846 1841 First Lord of the Treasury Leader of the House of Commons Conservative Peel II 7003176500000000000 ♠ 4 years and 304 days The Right Honourable Lord John Russell FRS MP for the City of London ( 1792 -- 1878 ) 30 June 1846 21 February 1852 1847 First Lord of the Treasury Leader of the House of Commons Whig Russell I 7003206300000000000 ♠ 5 years and 237 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 23 February 1852 17 December 1852 First Lord of the Treasury Leader of the House of Lords Conservative Who ? Who ? 1852 7002299000000000000 ♠ 299 days The Right Honourable George Hamilton - Gordon 4th Earl of Aberdeen KT FRSE FRS PC FSA Scot ( 1784 -- 1860 ) 19 December 1852 30 January 1855 -- First Lord of the Treasury Leader of the House of Lords Peelite Aberdeen ( Peelite -- Whig ) 7002773000000000000 ♠ 2 years and 43 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 6 February 1855 19 February 1858 First Lord of the Treasury Leader of the House of Commons Whig Palmerston I 1857 7003111000000000000 ♠ 3 years and 14 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby PC ( 1799 -- 1869 ) 20 February 1858 11 June 1859 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli II 7002477000000000000 ♠ 1 year and 112 days The Right Honourable Henry John Temple 3rd Viscount Palmerston KG GCB PC FRS MP for Tiverton ( 1784 -- 1865 ) 12 June 1859 18 October 1865 1859 First Lord of the Treasury Leader of the House of Commons Liberal Palmerston II 1865 7003232100000000000 ♠ 6 years and 129 days The Right Honourable John Russell 1st Earl Russell KG PC FRS ( 1792 -- 1878 ) 29 October 1865 26 June 1866 -- First Lord of the Treasury Leader of the House of Lords Liberal Russell II 7002241000000000000 ♠ 241 days The Right Honourable Edward Smith - Stanley 14th Earl of Derby KG PC ( 1799 -- 1869 ) 28 June 1866 25 February 1868 -- First Lord of the Treasury Leader of the House of Lords Conservative Derby -- Disraeli III 7002608000000000000 ♠ 1 year and 243 days The Right Honourable Benjamin Disraeli MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 -- 7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and 77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April 1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone II 23 April 1880 9 June 1885 1880 7003187400000000000 ♠ 5 years and 48 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 23 June 1885 28 January 1886 -- Sec. of State for Foreign Affairs Leader of the House of Lords Conservative Salisbury I 7002220000000000000 ♠ 220 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone III 1 February 1886 20 July 1886 1885 7002170000000000000 ♠ 170 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 July 1886 11 August 1892 1886 Leader of the House of Lords First Lord of the Treasury Sec. of State for Foreign Affairs Conservative Salisbury II 7003221000000000000 ♠ 6 years and 18 days The Right Honourable William Ewart Gladstone FRS FSS MP for Midlothian ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the Treasury Lord Keeper of the Privy Seal Leader of the House of Commons Liberal Gladstone IV 15 August 1892 2 March 1894 1892 7002565000000000000 ♠ 1 year and 200 days The Right Honourable Archibald Primrose 5th Earl of Rosebery KG PC FRS ( 1847 -- 1929 ) 5 March 1894 22 June 1895 -- First Lord of the Treasury Lord President of the Council Leader of the House of Lords Liberal Rosebery 7002475000000000000 ♠ 1 year and 110 days The Most Honourable Robert Gascoyne - Cecil 3rd Marquess of Salisbury KG PC FRS DL ( 1830 -- 1903 ) 25 June 1895 11 July 1902 Leader of the House of Lords Sec. of State for Foreign Affairs Lord Keeper of the Privy Seal Conservative Salisbury III ( Con. -- Lib. U . ) 1895 1900 7003257300000000000 ♠ 7 years and 17 days Edward VII ( 1901 -- 1910 ) The Right Honourable Arthur Balfour FRS DL MP for Manchester East ( 1848 -- 1930 ) 11 July 1902 5 December 1905 -- First Lord of the Treasury Leader of the House of Commons Conservative Balfour 7003124400000000000 ♠ 3 years and 148 days The Right Honourable Sir Henry Campbell - Bannerman GCB MP for Stirling Burghs ( 1836 -- 1908 ) 5 December 1905 7 April 1908 First Lord of the Treasury Leader of the House of Commons Liberal Campbell - Bannerman 1906 7002855000000000000 ♠ 2 years and 125 days The Right Honourable Herbert Henry Asquith KC FRS MP for East Fife ( 1852 -- 1928 ) 7 April 1908 5 December 1916 First Lord of the Treasury Leader of the House of Commons Sec. of State for War Liberal Asquith I 1910 ​ 1910 George V ( 1910 -- 1936 ) ​ Asquith II ( Lib. -- Con. -- et al . ) 7003316500000000000 ♠ 8 years and 243 days The Right Honourable David Lloyd George OM MP for Caernarvon Boroughs ( 1863 -- 1945 ) 6 December 1916 19 October 1922 First Lord of the Treasury Liberal Lloyd George War 1918 Lloyd George II ( Lib. -- Con . ) 7003214400000000000 ♠ 5 years and 318 days The Right Honourable Bonar Law MP for Glasgow Central ( 1858 -- 1923 ) 23 October 1922 20 May 1923 1922 First Lord of the Treasury Leader of the House of Commons Conservative ( Scot. U . ) Law 7002210000000000000 ♠ 210 days The Right Honourable Stanley Baldwin JP MP for Bewdley ( 1867 -- 1947 ) 23 May 1923 16 January 1924 -- First Lord of the Treasury Leader of the House of Commons Chancellor of the Exchequer Conservative Baldwin I 7002239000000000000 ♠ 239 days The Right Honourable Ramsay MacDonald MP for Aberavon ( 1866 -- 1937 ) 22 January 1924 4 November 1924 1923 First Lord of the Treasury Leader of the House of Commons Sec. of State for Foreign Affairs Labour MacDonald I 7002288000000000000 ♠ 288 days The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 4 November 1924 5 June 1929 1924 First Lord of the Treasury Leader of the House of Commons Conservative Baldwin II 7003167500000000000 ♠ 4 years and 214 days The Right Honourable Ramsay MacDonald FRS MP for Seaham ( 1866 -- 1937 ) 5 June 1929 7 June 1935 1929 First Lord of the Treasury Leader of the House of Commons Labour MacDonald II National Labour National I ( N. Lab. -- Con. -- et al . ) 1931 National II 7003219400000000000 ♠ 6 years and 3 days ​ The Right Honourable Stanley Baldwin JP FRS MP for Bewdley ( 1867 -- 1947 ) 7 June 1935 28 May 1937 1935 First Lord of the Treasury Leader of the House of Commons Conservative National III ( Con. -- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​ Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26 July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker ( Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26 July 1945 26 October 1951 1945 First Lord of the Treasury Minister of Defence Labour Attlee I 1950 Attlee II 7003228400000000000 ♠ 6 years and 93 days The Right Honourable Sir Winston Churchill KG OM CH TD PCc DL FRS RA MP for Woodford ( 1874 -- 1965 ) 26 October 1951 6 April 1955 1951 First Lord of the Treasury Minister of Defence Conservative Churchill III 7003125900000000000 ♠ 3 years and 163 days Elizabeth II ( 1952 -- present ) The Right Honourable Sir Anthony Eden KG MC MP for Warwick & Leamington ( 1897 -- 1977 ) 6 April 1955 10 January 1957 1955 First Lord of the Treasury Conservative Eden 7002646000000000000 ♠ 1 year and 280 days The Right Honourable Harold Macmillan FRS MP for Bromley ( 1894 -- 1986 ) 10 January 1957 19 October 1963 1959 First Lord of the Treasury Conservative Macmillan I ​ Macmillan II 7003247400000000000 ♠ 6 years and 283 days The Right Honourable Sir Alec Douglas - Home KT MP for Kinross & Western Perthshire ( 1903 -- 1995 ) 19 October 1963 16 October 1964 -- First Lord of the Treasury Conservative ( Scot. U . ) Douglas - Home 7002364000000000000 ♠ 364 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 16 October 1964 19 June 1970 1964 First Lord of the Treasury Minister for the Civil Service Labour Wilson I 1966 Wilson II 7003207300000000000 ♠ 5 years and 247 days The Right Honourable Edward Heath MBE MP for Bexley -- Sidcup ( 1916 -- 2005 ) 19 June 1970 4 March 1970 First Lord of the Treasury Minister for the Civil Service Conservative Heath 7003135500000000000 ♠ 3 years and 259 days The Right Honourable Harold Wilson OBE FRS MP for Huyton ( 1916 -- 1995 ) 4 March 5 April 1976 First Lord of the Treasury Minister for the Civil Service Labour Wilson III Wilson IV 7002764000000000000 ♠ 2 years and 33 days The Right Honourable James Callaghan MP for Cardiff South East ( 1912 -- 2005 ) 5 April 1976 4 May 1979 -- First Lord of the Treasury Minister for the Civil Service Labour Callaghan 7003112500000000000 ♠ 3 years and 30 days The Right Honourable Margaret Thatcher FRS FRIC MP for Finchley ( 1925 -- 2013 ) See also § Premierships :3 First Lord of the Treasury Minister for the Civil Service Conservative Thatcher I 4 May 1979 28 November 1990 1979 Thatcher II Thatcher III 7003422700000000000 ♠ 11 years and 209 days The Right Honourable John Major MP for Huntingdon ( born 1943 ) 28 November 1990 2 May 1997 First Lord of the Treasury Minister for the Civil Service Conservative Major I ​ Major II 7003234800000000000 ♠ 6 years and 156 days The Right Honourable Tony Blair MP for Sedgefield ( born 1953 ) See also § Premierships : 4 First Lord of the Treasury Minister for the Civil Service Labour Blair I 2 May 1997 27 June 2007 1997 Blair II 2005 Blair III 7003370900000000000 ♠ 10 years and 57 days The Right Honourable Gordon Brown MP for Kirkcaldy & Cowdenbeath ( born 1951 ) See also § Premierships : 5 First Lord of the Treasury Minister for the Civil Service Labour Brown 27 June 2007 11 May -- 7003105000000000000 ♠ 2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July 2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days
  Fact 0.0: Robert Walpole, 1st Earl of Orford, was MP for King's Lynn and lived from 1676 to 1745.
  Fact 0.1: Robert Walpole was Prime Minister from 4 April 1721 to 11 February 1742, for 20 years and 314 days.
  Fact 0.2: Robert Walpole was a Whig.
  Fact 0.3: Spencer Compton, 1st Earl of Wilmington, lived from 1673 to 1743.
  Fact 0.4: Spencer Compton was Prime Minister from 16 February 1742 to 2 July 1743, for 1 year and 137 days.
  Fact 0.5: Spencer Compton was a Whig.
  Fact 0.6: Henry Pelham was MP for Sussex and lived from 1694 to 1754.
  Fact 0.7: Henry Pelham was Prime Minister from 27 August 1743 to 6 March 1754, for 10 years and 192 days.
  Fact 0.8: Henry Pelham was a Whig.
  Fact 0.9: Thomas Pelham-Holles, 1st Duke of Newcastle, lived from 1693 to 1768.
  Fact 0.10: Thomas Pelham-Holles was Prime Minister from 16 March 1754 to 16 November 1756, for 2 years and 246 days.
  Fact 0.11: Thomas Pelham-Holles was a Whig.
  Fact 0.12: William Cavendish, 4th Duke of Devonshire, lived from 1720 to 1764.
  Fact 0.13: William Cavendish was Prime Minister from 16 November 1756 to 25 June 1757, for 222 days.
  Fact 0.14: William Cavendish's ministry was a caretaker ministry in 1757.
  Fact 0.15: William Cavendish was a Whig.
  Fact 0.16: Thomas Pelham-Holles was Prime Minister again from 2 July 1757 to 26 May 1762, for 4 years and 329 days.
  Fact 0.17: Thomas Pelham-Holles was a Whig in his second term.
  Fact 0.18: John Stuart, 3rd Earl of Bute, lived from 1713 to 1792.
  Fact 0.19: John Stuart was Prime Minister from 26 May 1762 to 8 April 1763, for 318 days.
  Fact 0.20: John Stuart was a Tory.
  Fact 0.21: George Grenville was MP for Buckingham and lived from 1712 to 1770.
  Fact 0.22: George Grenville was Prime Minister from 16 April 1763 to 13 July 1765, for 2 years and 89 days.
  Fact 0.23: George Grenville was a Whig, in the Grenvillite faction.
  Fact 0.24: Charles Watson-Wentworth, 2nd Marquess of Rockingham, lived from 1730 to 1782.
  Fact 0.25: Charles Watson-Wentworth was Prime Minister from 13 July 1765 to 30 July 1766, for 1 year and 18 days.
  Fact 0.26: Charles Watson-Wentworth was a Whig, in the Rockinghamite faction.
  Fact 0.27: William Pitt, 1st Earl of Chatham, was MP for Bath and lived from 1708 to 1778.
  Fact 0.28: William Pitt, 1st Earl of Chatham, was Prime Minister from 30 July 1766 to 14 October 1768, for 2 years and 77 days.
  Fact 0.29: William Pitt, 1st Earl of Chatham, was a Whig, in the Chathamite faction.
  Fact 0.30: Augustus FitzRoy, 3rd Duke of Grafton, lived from 1735 to 1811.
  Fact 0.31: Augustus FitzRoy was Prime Minister from 14 October 1768 to 28 January 1770, for 1 year and 107 days.
  Fact 0.32: Augustus FitzRoy was a Whig, in the Chathamite faction.
  Fact 0.33: Frederick North, Lord North, was MP for Banbury and lived from 1732 to 1792.
  Fact 0.34: Frederick North was Prime Minister from 28 January 1770 to 22 March 1782, for 12 years and 54 days.
  Fact 0.35: Frederick North was a Tory, in the Northite faction.
  Fact 0.36: Charles Watson-Wentworth was Prime Minister again from 27 March 1782 to 1 July 1782, for 97 days.
  Fact 0.37: Charles Watson-Wentworth was a Whig in his second term, in the Rockinghamite faction.
  Fact 0.38: William Petty, 2nd Earl of Shelburne, lived from 1737 to 1805.
  Fact 0.39: William Petty was Prime Minister from 4 July 1782 to 2 April 1783, for 273 days.
  Fact 0.40: William Petty was a Whig, in the Chathamite faction.
  Fact 0.41: William Cavendish-Bentinck, 3rd Duke of Portland, lived from 1738 to 1809.
  Fact 0.42: William Cavendish-Bentinck was Prime Minister from 2 April 1783 to 19 December 1783, for 262 days.
  Fact 0.43: William Cavendish-Bentinck was a Whig.
  Fact 0.44: William Pitt the Younger was MP for Appleby and Cambridge University, and lived from 1759 to 1806.
  Fact 0.45: William Pitt the Younger was Prime Minister from 19 December 1783 to 14 March 1801, for 17 years and 86 days.
  Fact 0.46: William Pitt the Younger was a Tory, in the Pittite faction.
  Fact 0.47: Henry Addington was MP for Devizes and lived from 1757 to 1844.
  Fact 0.48: Henry Addington was Prime Minister from 17 March 1801 to 10 May 1804, for 3 years and 55 days.
  Fact 0.49: Henry Addington was a Tory, in the Pittite faction.
  Fact 0.50: William Pitt the Younger was Prime Minister again from 10 May 1804 to 23 January 1806, for 1 year and 259 days.
  Fact 0.51: William Pitt the Younger was a Tory, in the Pittite faction, in his second term.
  Fact 0.52: William Grenville, 1st Baron Grenville, lived from 1759 to 1834.
  Fact 0.53: William Grenville was Prime Minister from 11 February 1806 to 31 March 1807, for 1 year and 49 days.
  Fact 0.54: William Grenville was a Whig, in the All the Talents ministry.
  Fact 0.55: William Cavendish-Bentinck was Prime Minister again from 31 March 1807 to 4 October 1809, for 2 years and 188 days.
  Fact 0.56: William Cavendish-Bentinck was a Tory, in the Pittite faction, in his second term.
  Fact 0.57: Spencer Perceval was MP for Northampton and lived from 1762 to 1812.
  Fact 0.58: Spencer Perceval was Prime Minister from 4 October 1809 to 11 May 1812, for 2 years and 221 days.
  Fact 0.59: Spencer Perceval was a Tory, in the Pittite faction.
  Fact 0.60: Robert Jenkinson, 2nd Earl of Liverpool, lived from 1770 to 1828.
  Fact 0.61: Robert Jenkinson was Prime Minister from 8 June 1812 to 9 April 1827, for 14 years and 306 days.
  Fact 0.62: Robert Jenkinson was a Tory, in the Pittite faction.
  Fact 0.63: George Canning was MP for Seaford and lived from 1770 to 1827.
  Fact 0.64: George Canning was Prime Minister from 10 April 1827 to 8 August 1827, for 121 days.
  Fact 0.65: George Canning was a Tory, in the Canningite faction.
  Fact 0.66: Frederick John Robinson, 1st Viscount Goderich, lived from 1782 to 1859.
  Fact 0.67: Frederick John Robinson was Prime Minister from 31 August 1827 to 21 January 1828, for 144 days.
  Fact 0.68: Frederick John Robinson was a Tory, in the Canningite faction.
  Fact 0.69: Arthur Wellesley, 1st Duke of Wellington, lived from 1769 to 1852.
  Fact 0.70: Arthur Wellesley was Prime Minister from 22 January 1828 to 16 November 1830, for 2 years and 299 days.
  Fact 0.71: Arthur Wellesley was a Tory.
  Fact 0.72: Charles Grey, 2nd Earl Grey, lived from 1764 to 1845.
  Fact 0.73: Charles Grey was Prime Minister from 22 November 1830 to 9 July 1834, for 3 years and 230 days.
  Fact 0.74: Charles Grey was a Whig.
  Fact 0.75: William Lamb, 2nd Viscount Melbourne, lived from 1779 to 1848.
  Fact 0.76: William Lamb was Prime Minister from 16 July 1834 to 14 November 1834, for 122 days.
  Fact 0.77: William Lamb was a Whig.
  Fact 0.78: Arthur Wellesley was Prime Minister again from 14 November 1834 to 10 December 1834, for 27 days.
  Fact 0.79: Arthur Wellesley's second term was a caretaker ministry, and he was a Tory.
  Fact 0.80: Robert Peel was MP for Tamworth and lived from 1788 to 1850.
  Fact 0.81: Robert Peel was Prime Minister from 10 December 1834 to 8 April 1835, for 120 days.
  Fact 0.82: Robert Peel was a Conservative.
  Fact 0.83: William Lamb was Prime Minister again from 18 April 1835 to 30 August 1841, for 6 years and 135 days.
  Fact 0.84: William Lamb was a Whig in his second term.
  Fact 0.85: Robert Peel was Prime Minister again from 30 August 1841 to 29 June 1846, for 4 years and 304 days.
  Fact 0.86: Robert Peel was a Conservative in his second term.
  Fact 0.87: Lord John Russell was MP for the City of London and lived from 1792 to 1878.
  Fact 0.88: Lord John Russell was Prime Minister from 30 June 1846 to 21 February 1852, for 5 years and 237 days.
  Fact 0.89: Lord John Russell was a Whig.
  Fact 0.90: Edward Smith-Stanley, 14th Earl of Derby, lived from 1799 to 1869.
  Fact 0.91: Edward Smith-Stanley was Prime Minister from 23 February 1852 to 17 December 1852, for 299 days.
  Fact 0.92: Edward Smith-Stanley was a Conservative.
  Fact 0.93: George Hamilton-Gordon, 4th Earl of Aberdeen, lived from 1784 to 1860.
  Fact 0.94: George Hamilton-Gordon was Prime Minister from 19 December 1852 to 30 January 1855, for 2 years and 43 days.
  Fact 0.95: George Hamilton-Gordon was a Peelite.
  Fact 0.96: Henry John Temple, 3rd Viscount Palmerston, was MP for Tiverton and lived from 1784 to 1865.
  Fact 0.97: Henry John Temple was Prime Minister from 6 February 1855 to 19 February 1858, for 3 years and 14 days.
  Fact 0.98: Henry John Temple was a Whig.
  Fact 0.99: Edward Smith-Stanley was Prime Minister again from 20 February 1858 to 11 June 1859, for 1 year and 112 days.
  Fact 0.100: Edward Smith-Stanley was a Conservative in his second term.
  Fact 0.101: Henry John Temple was Prime Minister again from 12 June 1859 to 18 October 1865, for 6 years and 129 days.
  Fact 0.102: Henry John Temple was a Liberal in his second term.
  Fact 0.103: John Russell, 1st Earl Russell, lived from 1792 to 1878.
  Fact 0.104: John Russell, 1st Earl Russell, was Prime Minister from 29 October 1865 to 26 June 1866, for 241 days.
  Fact 0.105: John Russell, 1st Earl Russell, was a Liberal.
  Fact 0.106: Edward Smith-Stanley was Prime Minister again from 28 June 1866 to 25 February 1868, for 1 year and 243 days.
  Fact 0.107: Edward Smith-Stanley was a Conservative in his third term.
  Fact 0.108: Benjamin Disraeli was MP for Buckinghamshire and lived from 1804 to 1881.
  Fact 0.109: Benjamin Disraeli was Prime Minister from 27 February 1868 to 1 December 1868, for 279 days.
  Fact 0.110: Benjamin Disraeli was a Conservative.
  Fact 0.111: William Ewart Gladstone was MP for Greenwich and lived from 1809 to 1898.
  Fact 0.112: William Ewart Gladstone was Prime Minister from 3 December 1868 to 17 February 1874, for 5 years and 77 days.
  Fact 0.113: William Ewart Gladstone was a Liberal.
  Fact 0.114: Benjamin Disraeli, 1st Earl of Beaconsfield, was Prime Minister again from 20 February 1874 to 21 April 1880, for 6 years and 62 days.
  Fact 0.115: Benjamin Disraeli was a Conservative in his second term.
  Fact 0.116: William Ewart Gladstone was MP for Midlothian and was Prime Minister again from 23 April 1880 to 9 June 1885, for 5 years and 48 days.
  Fact 0.117: William Ewart Gladstone was a Liberal in his second term.
  Fact 0.118: Robert Gascoyne-Cecil, 3rd Marquess of Salisbury, lived from 1830 to 1903.
  Fact 0.119: Robert Gascoyne-Cecil was Prime Minister from 23 June 1885 to 28 January 1886, for 220 days.
  Fact 0.120: Robert Gascoyne-Cecil was a Conservative.
  Fact 0.121: William Ewart Gladstone was Prime Minister again from 1 February 1886 to 20 July 1886, for 170 days.
  Fact 0.122: William Ewart Gladstone was a Liberal in his third term.
  Fact 0.123: Robert Gascoyne-Cecil was Prime Minister again from 25 July 1886 to 11 August 1892, for 6 years and 18 days.
  Fact 0.124: Robert Gascoyne-Cecil was a Conservative in his second term.
  Fact 0.125: William Ewart Gladstone was Prime Minister again from 15 August 1892 to 2 March 1894, for 1 year and 200 days.
  Fact 0.126: William Ewart Gladstone was a Liberal in his fourth term.
  Fact 0.127: Archibald Primrose, 5th Earl of Rosebery, lived from 1847 to 1929.
  Fact 0.128: Archibald Primrose was Prime Minister from 5 March 1894 to 22 June 1895, for 1 year and 110 days.
  Fact 0.129: Archibald Primrose was a Liberal.
  Fact 0.130: Robert Gascoyne-Cecil was Prime Minister again from 25 June 1895 to 11 July 1902, for 7 years and 17 days.
  Fact 0.131: Robert Gascoyne-Cecil's third term was labelled Con. and Lib. U. in the table.
  Fact 0.132: Arthur Balfour was MP for Manchester East and lived from 1848 to 1930.
  Fact 0.133: Arthur Balfour was Prime Minister from 11 July 1902 to 5 December 1905, for 3 years and 148 days.
  Fact 0.134: Arthur Balfour was a Conservative.
  Fact 0.135: Henry Campbell-Bannerman was MP for Stirling Burghs and lived from 1836 to 1908.
  Fact 0.136: Henry Campbell-Bannerman was Prime Minister from 5 December 1905 to 7 April 1908, for 2 years and 125 days.
  Fact 0.137: Henry Campbell-Bannerman was a Liberal.
  Fact 0.138: Herbert Henry Asquith was MP for East Fife and lived from 1852 to 1928.
  Fact 0.139: Herbert Henry Asquith was Prime Minister from 7 April 1908 to 5 December 1916, for 8 years and 243 days.
  Fact 0.140: Herbert Henry Asquith was a Liberal.
  Fact 0.141: David Lloyd George was MP for Caernarvon Boroughs and lived from 1863 to 1945.
  Fact 0.142: David Lloyd George was Prime Minister from 6 December 1916 to 19 October 1922, for 5 years and 318 days.
  Fact 0.143: David Lloyd George was a Liberal.
  Fact 0.144: Bonar Law was MP for Glasgow Central and lived from 1858 to 1923.
  Fact 0.145: Bonar Law was Prime Minister from 23 October 1922 to 20 May 1923, for 210 days.
  Fact 0.146: Bonar Law was a Conservative.
  Fact 0.147: Stanley Baldwin was MP for Bewdley and lived from 1867 to 1947.
  Fact 0.148: Stanley Baldwin was Prime Minister from 23 May 1923 to 16 January 1924, for 239 days.
  Fact 0.149: Stanley Baldwin was a Conservative.
  Fact 0.150: Ramsay MacDonald was MP for Aberavon and lived from 1866 to 1937.
  Fact 0.151: Ramsay MacDonald was Prime Minister from 22 January 1924 to 4 November 1924, for 288 days.
  Fact 0.152: Ramsay MacDonald was a Labour politician.
  Fact 0.153: Stanley Baldwin was Prime Minister again from 4 November 1924 to 5 June 1929, for 4 years and 214 days.
  Fact 0.154: Stanley Baldwin was a Conservative in his second term.
  Fact 0.155: Ramsay MacDonald was MP for Seaham and was Prime Minister again from 5 June 1929 to 7 June 1935, for 6 years and 3 days.
  Fact 0.156: Ramsay MacDonald was a Labour politician in his second term.
  Fact 0.157: Stanley Baldwin was Prime Minister again from 7 June 1935 to 28 May 1937, for 1 year and 356 days.
  Fact 0.158: Stanley Baldwin was a Conservative in his third term.
  Fact 0.159: Neville Chamberlain was MP for Birmingham Edgbaston and lived from 1869 to 1940.
  Fact 0.160: Neville Chamberlain was Prime Minister from 28 May 1937 to 10 May 1940, for 2 years and 349 days.
  Fact 0.161: Neville Chamberlain was a Conservative.
  Fact 0.162: Winston Churchill was MP for Epping and lived from 1874 to 1965.
  Fact 0.163: Winston Churchill was Prime Minister from 10 May 1940 to 26 July 1945, for 5 years and 78 days.
  Fact 0.164: Winston Churchill was a Conservative.
  Fact 0.165: Clement Attlee was MP for Limehouse and Walthamstow West and lived from 1883 to 1967.
  Fact 0.166: Clement Attlee was Prime Minister from 26 July 1945 to 26 October 1951, for 6 years and 93 days.
  Fact 0.167: Clement Attlee was a Labour politician.
  Fact 0.168: Winston Churchill was MP for Woodford and was Prime Minister again from 26 October 1951 to 6 April 1955, for 3 years and 163 days.
  Fact 0.169: Winston Churchill was a Conservative in his second term.
  Fact 0.170: Anthony Eden was MP for Warwick and Leamington and lived from 1897 to 1977.
  Fact 0.171: Anthony Eden was Prime Minister from 6 April 1955 to 10 January 1957, for 1 year and 280 days.
  Fact 0.172: Anthony Eden was a Conservative.
  Fact 0.173: Harold Macmillan was MP for Bromley and lived from 1894 to 1986.
  Fact 0.174: Harold Macmillan was Prime Minister from 10 January 1957 to 19 October 1963, for 6 years and 283 days.
  Fact 0.175: Harold Macmillan was a Conservative.
  Fact 0.176: Alec Douglas-Home was MP for Kinross and Western Perthshire and lived from 1903 to 1995.
  Fact 0.177: Alec Douglas-Home was Prime Minister from 19 October 1963 to 16 October 1964, for 364 days.
  Fact 0.178: Alec Douglas-Home was a Conservative, in the Scottish Unionist faction.
  Fact 0.179: Harold Wilson was MP for Huyton and lived from 1916 to 1995.
  Fact 0.180: Harold Wilson was Prime Minister from 16 October 1964 to 19 June 1970, for 5 years and 247 days.
  Fact 0.181: Harold Wilson was a Labour politician.
  Fact 0.182: Edward Heath was MP for Bexley and Sidcup and lived from 1916 to 2005.
  Fact 0.183: Edward Heath was Prime Minister from 19 June 1970 to 4 March 1970, as the table lists it, for 3 years and 259 days.
  Fact 0.184: Edward Heath was a Conservative.
  Fact 0.185: Harold Wilson was Prime Minister again from 4 March to 5 April 1976, for 2 years and 33 days.
  Fact 0.186: Harold Wilson was a Labour politician in his second term.
  Fact 0.187: James Callaghan was MP for Cardiff South East and lived from 1912 to 2005.
  Fact 0.188: James Callaghan was Prime Minister from 5 April 1976 to 4 May 1979, for 3 years and 30 days.
  Fact 0.189: James Callaghan was a Labour politician.
  Fact 0.190: Margaret Thatcher was MP for Finchley and lived from 1925 to 2013.
  Fact 0.191: Margaret Thatcher was Prime Minister from 4 May 1979 to 28 November 1990, for 11 years and 209 days.
  Fact 0.192: Margaret Thatcher was a Conservative.
  Fact 0.193: John Major was MP for Huntingdon and was born in 1943.
  Fact 0.194: John Major was Prime Minister from 28 November 1990 to 2 May 1997, for 6 years and 156 days.
  Fact 0.195: John Major was a Conservative.
  Fact 0.196: Tony Blair was MP for Sedgefield and was born in 1953.
  Fact 0.197: Tony Blair was Prime Minister from 2 May 1997 to 27 June 2007, for 10 years and 57 days.
  Fact 0.198: Tony Blair was a Labour politician.
  Fact 0.199: Gordon Brown was MP for Kirkcaldy and Cowdenbeath and was born in 1951.
  Fact 0.200: Gordon Brown was Prime Minister from 27 June 2007 to 11 May (the year is not shown in the table), for 2 years and 319 days.
  Fact 0.201: Gordon Brown was a Labour politician.
  Fact 0.202: David Cameron was MP for Witney and was born in 1966.
  Fact 0.203: David Cameron was Prime Minister from 11 May (the year is not shown in the table) to 13 July 2016, for 6 years and 64 days.
  Fact 0.204: David Cameron was a Conservative, in a coalition with the Liberal Democrats.
  Fact 0.205: Theresa May was MP for Maidenhead and was born in 1956.
  Fact 0.206: Theresa May has been Prime Minister since 13 July 2016.
  Fact 0.207: Theresa May's term was 1 year and 193 days as of the table.
  Fact 0.208: Theresa May was a Conservative.
  Fact 0.209: The monarch Elizabeth II reigned from 1952 to the present.
  Fact 0.210: The monarch George I reigned from 1714 to 1727.
  Fact 0.211: The monarch George II reigned from 1727 to 1760.
  Fact 0.212: The monarch George III reigned from 1760 to 1820.
  Fact 0.213: The monarch George IV reigned from 1820 to 1830.
  Fact 0.214: The monarch William IV reigned from 1830 to 1837.
  Fact 0.215: The monarch Victoria reigned from 1837 to 1901.
  Fact 0.216: The monarch Edward VII reigned from 1901 to 1910.
  Fact 0.217: The monarch George V reigned from 1910 to 1936.
  Fact 0.218: The monarch Edward VIII reigned in 1936.
  Fact 0.219: The monarch George VI reigned from 1936 to 1952.

Retrieved items:
<item 1>
List of Prime Ministers of the United Kingdom
List of Prime Ministers of the United Kingdom - wikipedia List of Prime Ministers of the United Kingdom Jump to : navigation , search Top left : Walpole became
the first Prime Minister of the Kingdom of Great Britain in 1721 . Top right : Pitt the Younger became the first Prime Minister of the United Kingdom of Great
Britain and Ireland in 1801 . Bottom left : Lloyd George was the last Prime Minister of the United Kingdom of Great Britain and Ireland as a result of the Anglo
- Irish Treaty in 1922 . Bottom right : Baldwin was Prime Minister when the country was renamed the United Kingdom of Great Britain and Northern Ireland in 1927
. The Prime Minister of the United Kingdom is the head of the Government of the United Kingdom , and chairs Cabinet meetings . There is no specific date when
the office of Prime Minister first appeared , as the role was not created but rather evolved over a period of time . The term was used in the House of Commons
in 1805 and it was certainly in parliamentary use by the 1880s , and in 1905 the post of Prime Minister was officially given recognition in the order of
precedence .
</item 1>

<item 2>
List of Prime Ministers of the United Kingdom
77 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 85 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 94 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 98 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 101 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 106 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 116 . Jump up ^
Eccleshall & Walker ( 2002 ) , p. 120 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 123 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 128 . ^ Jump up to :
Eccleshall & Walker ( 2002 ) , p. 136 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 142 .
</item 2>

<item 3>
List of Prime Ministers of the United Kingdom
^ 7 Majority Conservative government before dissolution . See also ( edit ) United Kingdom This article is part of a series on the politics and government of
the United Kingdom Constitution ( show ) Constitution Cabinet Manual Erskine May Taxation The Crown ( show ) British Monarchy Monarch Queen Elizabeth II Heir
apparent The Prince of Wales Royal family Succession to the British throne Royal prerogative Executive ( show ) HM Government Prime Minister Theresa May Cabinet
Current ministry Her Majesty 's Civil Service Departments Executive agencies Public bodies Privy Council Crown - in - Council Queen Elizabeth II Legislature (
show ) Parliament ( 56th Parliament ) Crown - in - Parliament Queen Elizabeth II House of Lords Lord Speaker The Lord Fowler House of Commons Speaker John
Bercow Leader of the Opposition Jeremy Corbyn Her Majesty 's Loyal Opposition Shadow Cabinet State Opening of Parliament Prime Minister 's Questions Act of
Parliament Statutory Instruments Judiciary ( show ) Supreme Court President The Baroness Hale of Richmond Deputy President The Lord Mance Judges of the Supreme
Court Courts of the United Kingdom Courts of England and Wales Courts of Scotland Courts of Northern Ireland Devolution ( show ) Joint Ministerial Committee
Scotland Government Parliament Wales Government National Assembly Reserved matters Northern Ireland Executive Assembly Administrative geography ( show )
</item 3>

<item 4>
List of Prime Ministers of the United Kingdom
-- 1881 ) See also § Premierships : 1 First Lord of the Treasury Leader of the House of Commons Conservative 27 February 1868 1 December 1868 --
7002279000000000000 ♠ 279 days The Right Honourable William Ewart Gladstone FSS MP for Greenwich ( 1809 -- 1898 ) See also § Premierships : 2 First Lord of the
Treasury Leader of the House of Commons Chancellor of the Exchequer Liberal Gladstone I 3 December 1868 17 February 1874 1868 7003190300000000000 ♠ 5 years and
77 days The Right Honourable Benjamin Disraeli 1st Earl of Beaconsfield KG PC FRS MP for Buckinghamshire ( 1804 -- 1881 ) See also § Premierships : 1 First Lord
of the Treasury Leader of the House of Commons Lord Keeper of the Privy Seal Leader of the House of Lords Conservative Disraeli II 20 February 1874 21 April
1880 1874 7003225300000000000 ♠ 6 years and 62 days The Right Honourable William
</item 4>

<item 5>
List of Prime Ministers of the United Kingdom
14 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 17 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 28 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 34 .
Jump up ^ Eccleshall & Walker ( 2002 ) , p. 36 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 42 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 46 . Jump up
^ Eccleshall & Walker ( 2002 ) , p. 54 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 61 . Jump up ^ Eccleshall & Walker ( 2002 ) , p. 64 . Jump up ^ Eccleshall
& Walker ( 2002 ) , p. 73 . ^ Jump up to : Eccleshall & Walker ( 2002 ) , p. 77 .
</item 5>

<item 6>
List of Prime Ministers of the United Kingdom
-- N. Lab. -- et al . ) Edward VIII ( 1936 ) George VI ( 1936 -- 1952 ) 7002722000000000000 ♠ 1 year and 356 days The Right Honourable Neville Chamberlain FRS
MP for Birmingham Edgbaston ( 1869 -- 1940 ) 28 May 1937 10 May 1940 -- First Lord of the Treasury Leader of the House of Commons Conservative National IV ​
Chamberlain War 7003107900000000000 ♠ 2 years and 349 days The Right Honourable Winston Churchill CH TD PCc FRS MP for Epping ( 1874 -- 1965 ) 10 May 1940 26
July 1945 -- First Lord of the Treasury Minister of Defence Leader of the House of Commons Conservative Churchill War ( All parties ) ​ Churchill Caretaker (
Con. -- Lib. N . ) 7003190400000000000 ♠ 5 years and 78 days The Right Honourable Clement Attlee CH FRS MP for Limehouse -- Walthamstow West ( 1883 -- 1967 ) 26
July
</item 6>

<item 7>
List of Prime Ministers of the United Kingdom
BBC News . 13 July 2016 . Retrieved 26 July 2016 . Lay summary ( 13 July 2016 ) . Bibliography ( edit ) Burt , Llewellyn C. ( 1874 ) . A synoptical history of
England ( 2nd ed . ) . London : Lockwood . Carpenter , Clive , ed. ( 1992 ) . The Guinness UK Data Book . Guinness . ISBN 978 - 0 - 85112 - 522 - 0 . Chapman ,
Richard A. ( 31 January 2002 ) . `` History : from earliest times to the present day '' . The Treasury in Public Policy - Making . Routledge . ISBN 978 - 1 -
134 - 86426 - 3 . Clarke , John ( 1999 ) . `` House of Hanover '' . In Fraser , Antonia . The Lives of the Kings & Queens of England . University of California
Press ( published February 2000 ) .
</item 7>

<item 8>
List of Prime Ministers of the United Kingdom
After the succession of King George I in 1714 , the arrangement of a commission of Lords of the Treasury as opposed to a single Lord High Treasurer became
permanent . For the next three years , the government was headed by Lord Townshend , who was appointed Secretary of State for the Northern Department .
Subsequently , Lord Stanhope and Lord Sunderland ran the government jointly , with Stanhope managing foreign affairs and Sunderland managing domestic affairs .
Stanhope died in February 1721 and Sunderland resigned two months later ; Townshend and Robert Walpole were then invited to form the next government . From that
point , the holder of the office of First Lord also unofficially held the status of Prime Minister . It was not until the Edwardian era that the title Prime
Minister was constitutionally recognised . With only few exceptions , the Prime Minister still holds the office of First Lord by constitutional convention .
Since 1721 ( edit ) By century and monarch 18th to 19th century George I George II George III George IV William IV Victoria 20th to 21st century Edward VII
George V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative (
</item 8>

<item 9>
List of Prime Ministers of the United Kingdom
2 years and 319 days The Right Honourable David Cameron MP for Witney ( born 1966 ) See also § Premierships : 6 First Lord of the Treasury Minister for the
Civil Service Conservative Cameron -- Clegg ( Con. -- L.D. ) 11 May 13 July 2016 2015 Cameron II 7003225600000000000 ♠ 6 years and 64 days The Right Honourable
Theresa May MP for Maidenhead ( born 1956 ) See also § Premierships : 7 First Lord of the Treasury Minister for the Civil Service Conservative May I 13 July
2016 Incumbent 2017 May II 7002558000000000000 ♠ 1 year and 193 days Addenda ( edit ) ^ 1 Whig majority in the Commons before the dissolution of Parliament . ^
2 Minority government before dissolution . ^ 3 Tory majority in the Commons before dissolution . ^ 4 Majority Tory government before dissolution . ^ 5 Majority
Liberal government before dissolution . ^ 6 Coalition government during the course of an undissolved hung parliament . ^ 7 Majority Conservative government
before dissolution . See
</item 9>

<item 10>
List of Prime Ministers of the United Kingdom
V Edward VIII George VI Elizabeth II Whig ( 17 ) Tory ( 11 ) Conservative ( 17 ) Peelite ( 1 ) Liberal ( 7 ) Labour ( 6 ) National Labour ( 1 ) Portrait Name (
Birth -- Death ) Term of office Mandate Ministerial portfolios held as Prime Minister Party Ministry Monarch ( Reign ) Duration The Right Honourable Robert
Walpole 1st Earl of Orford KG KB PC MP for King 's Lynn ( 1676 -- 1745 ) 4 April 1721 11 February 1742 1722 First Lord of the Treasury Chancellor of the
Exchequer Leader of the House of Commons Whig Walpole -- Townshend George I ( 1714 -- 1727 ) 1727 George II ( 1727 -- 1760 ) 1734 Walpole 1741
7003761900000000000 ♠ 20 years and 314 days The Right Honourable Spencer Compton 1st Earl of Wilmington KG KB PC ( 1673 -- 1743 ) 16 February 1742 2 July 1743
-- First Lord of the Treasury Whig Carteret
</item 10>
=== END ITEM id=811de12e0db83626 ===
