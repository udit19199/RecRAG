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

This file has 3 items.

=== ITEM id=34b112e2ffc37a50 ===
Gold sentences and facts:
Sentence 0: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.0: The guide is a Diablo 4 guide.
  Fact 0.1: The guide gathers the best Diablo 4 Sorcerer builds for season 2.
  Fact 0.2: The guide simplifies the best Diablo 4 Sorcerer builds for season 2.
Sentence 1: In this Diablo 4 guide, we’ve gathered and simplified the best Diablo 4 Barbarian builds for season 2.
  Fact 1.0: The guide is a Diablo 4 guide.
  Fact 1.1: The guide gathers the best Diablo 4 Barbarian builds for season 2.
  Fact 1.2: The guide simplifies the best Diablo 4 Barbarian builds for season 2.

Retrieved items:
<item 1>
Comeback kings Richmond take the AFLW points in Cairns (The Age, 2023-10-20T23:51:34+00:00)
It was the Western Bulldogs’ eighth loss in a row, and their biggest-ever defeat, and they have just two rounds left to clinch a win this season. They next play
17th-placed West Coast and then top-four side North Melbourne. For the Swans, however, the victory has them challenging for a top-eight spot alongside the likes
of Gold Coast, Geelong, Collingwood, St Kilda and Carlton.

The Swans, who went winless in their debut season last year, were propelled to victory by a Chloe Molloy masterclass. After a five-goal second term, which
included a four-minute burst of three majors, Sydney had already brought up their largest score in an AFLW game by halftime. Swan Chloe Molloy celebrates a
goal. Credit: AFL Photos / Getty Images The Bulldogs held their own in the last quarter, but the damage had already been done. The total was also the biggest
the Dogs have conceded in their history, topping the 83 Melbourne kicked against them in round three.
</item 1>

<item 2>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
With only 0.4 of a percentage point separating Sydney in ninth and St Kilda in eighth, the win assured the Swans of retaking eighth spot. If the Saints had
qualified, it would have been their maiden finals appearance. Saints goalkicker Jesse Wardlaw had a heavy collision with Carlton’s Keeley Sherar late in the
final quarter of their clash.

Georgia Patrikios showed her class in the do-or-die match, collecting 25 disposals and eight contested possessions in a powerhouse display, while Hannah Priest
was a brick wall in defence, with nine intercept possessions. Abbie McKay worked overtime to counter Patrikios in the midfield, but her side was outclassed
across the park. McKay racked up 28 disposals and 12 clearances alongside Mimi Hill (30 disposals), while defender Sherar (17) toiled against a potent Saints
attack. Meanwhile, Sydney prevailed in a fierce tug-of-war with the Dockers to seal their first AFLW finals appearance.
</item 2>

<item 3>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But more modest attendances in subsequent games, including two sub-2,000 crowds in Perth, suggest the popularity of the Matildas’ run to the semi-finals has
only had a limited impact on support for domestic women’s sport.

skip past newsletter promotion Sign up to Australia Sport Free newsletter Get a daily roundup of the latest sports news, features and comment from our
Australian sports desk Privacy Notice: Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see
our Newsletters may contain info about charities, online ads, and content funded by outside parties. For more information see our Privacy Policy . We use Google
reCaptcha to protect our website and the Google Privacy Policy and Terms of Service apply. after newsletter promotion

Livingstone said the “green and gold fever” associated with the Women’s World Cup was a one-off and the AFL has plans in place for long-term growth.
</item 3>

<item 4>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“We’ll continue to keep building the base and keep building the foundations and have it not just strong at year seven, season eight, but strong for generations
to come.”

In a bid to improve the spectacle of the AFLW, the league introduced longer quarters this year and an interchange cap to help increase scoring. The effect,
according to Livingstone, has been positive.

“We’ve increased game time by eight minutes per match, on average, and we’ve had record scores across the rounds,” she said. “Whereas leading into the season we
hadn’t scored more than 100 goals in a round, we’ve regularly been scoring over 100 goals in a round.”

The Swans recorded their highest total last round, in their 14-12-96 to 6-3-39 defeat of the Western Bulldogs, the league’s winless bottom side. Molloy recorded
17 score involvements to set a new AFLW mark.

But Livingstone said the AFL was happy with the level of competition.
</item 4>

<item 5>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
WEEK ONE FINALS FIXTURE SATURDAY NOVEMBER 11 First Qualifying Final: Adelaide v Brisbane Lions, 2.35pm local (3.05pm AEDT), Norwood Oval First Elimination
Final: Gold Coast v Sydney, 6.15pm local (7.15pm AEDT), Heritage Bank Stadium SUNDAY NOVEMBER 12 Second Elimination Final: Geelong v Essendon, 1.05pm AEDT,
GMHBA Stadium Second Qualifying Final: Melbourne v North Melbourne, 3.05pm AEDT, Ikon Park The eight was not decided until the final game of the regular season,
when Sydney downed Fremantle by 22 points to leapfrog St Kilda into the final spot, after the Saints had kept themselves in the race with a 20-point win over
Carlton earlier in the day, 7.4 (46) to 3.8 (26).
</item 5>

<item 6>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But crowds are short of targets that could accelerate the extension of the regular season, which grows from 10 games this year to 12 games by 2025.

5,474 fans were at North Sydney Oval to watch the Swans’ first AFLW victory earlier in the season. Photograph: Jack Snape/The Guardian

The league enjoyed strong attendances in round one, which were partly attributed to the so-called Matildas effect, but some clubs have struggled to attract fans
in large numbers this season. Crowds of less than 1,500 have been recorded in both traditional markets such as Melbourne and Perth, as well as areas targeted
for growth like Blacktown and the Gold Coast.

The A-League Women launched this month with a first round highlighted by a crowd of more than 11,000 in Sydney.
</item 6>

<item 7>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
Quince paste and hand-picked grapes have played a part in helping the Sydney Swans launch themselves to the top of the AFLW attendance charts, less than two
years after their first match in the women’s league.

The Swans play their final home match of the regular season this weekend with an average attendance of 4,365 from four games so far – 124 more than the next
best club Adelaide – and a marquee pride match against Collingwood at Henson Park to come on Sunday where a crowd in excess of 4,000 is expected.

The club’s focus has been to target local communities and make match days compelling for fans on and off the field, according to executive GM of marketing at
the Swans, Loretta Johns.

“Our team has put a lot of work into sourcing some wonderful food offerings, entertainment and activities for kids, while we’ve had a local focus in promoting
our games,” she said.

Fans of Australian rules football in the Harbour City have a reputation for being fickle.
</item 7>

<item 8>
Ground-breaking Gold Coast: Suns make AFLW finals for a second time (The Age, 2023-11-03T23:45:04+00:00)
Down by four points at half-time in the low-scoring game, a third-term goal from Rebecca Privitelli sparked the Swans to a 5.10 (40) to 2.6 (18) win on Sunday
at Fremantle Oval.

The second-year expansion side stormed home with three unanswered goals to secure eighth spot. It’s a dramatic improvement, with the Swans winless in their
first season. “Any club that’s able to go from winless one year to a finals berth, it’s incredible,” Sydney co-captain Chloe Molloy told Fox Footy after the
match. “You can see the emotion on the girls’ faces. Coming into the season, the odds were against us. Just to get into the eight, we’ve managed to do that.
We’re stoked. It’s pretty special for the club.”
</item 8>

<item 9>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
“Even at Henson Park [the Swans’ other home ground in Sydney’s gentrified inner west], they’ve gone to microbreweries in the local area and invited them in, but
also promoted through their clientele to come to the football.”

The Swans forward Bella Smith said the players were proud of the club’s attendance record, and were grateful for the support. “We get there two hours before the
start of the game and the hill [at Henson Park] is bare. When you come out for the first team warm-up, the hill is slowly filling and then by the time the first
bounce gets around, it is full,” she said. “You can hear it, the atmosphere is amazing.”

The average attendance for the AFLW is a little over 2,500 a match this year, tracking approximately on par with previous seasons and with a finals uptick
likely to come.
</item 9>

<item 10>
Craft beer and cheese boxes: Swans tap into something ‘uniquely Sydney’ to draw AFLW crowds | Jack Snape (The Guardian, 2023-10-26T23:42:29+00:00)
But Livingstone said the AFL was happy with the level of competition. “Our ladder, from position five to 13 – there’s only a game separating it. It just shows
that this is a really competitive competition.”
</item 10>
=== END ITEM id=34b112e2ffc37a50 ===

=== ITEM id=e30baef8283a3419 ===
Gold sentences and facts:
Sentence 0: But by the age of 16, she is public property – incapable of going outside without being mobbed, and her love of singing and dancing now a lucrative resource.
  Fact 0.0: By the age of 16, the person described is public property.
  Fact 0.1: By the age of 16, the person described is incapable of going outside without being mobbed.
  Fact 0.2: By the age of 16, the person described's love of singing and dancing is now a lucrative resource.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker described it as almost a very strange, subtle clue to the media.
  Fact 1.1: The clue signals that the media does not know everything that happened in that relationship.
  Fact 1.2: The speaker says something really major and traumatic could have happened to her that the media does not know about.
  Fact 1.3: The speaker told NPR this.

Retrieved items:
<item 1>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Britney Spears and Justin Timberlake in 2002. Photograph: Tom Mihalek/AFP/Getty Images

When Timberlake tells the world that she cheated on him, she is all but emblazoned with a scarlet letter. While she knows she is powerless against that
narrative (“I don’t think Justin realised the power he had in shaming me. I don’t think he understands to this day”), him telling the world that they had a
sexual relationship was liberating. “To be honest with you, I liked that Justin said that,” she writes. “Why did my managers work so hard to claim I was some
kind of young-girl virgin even into my 20s? Whose business was it if I’d had sex or not?”

But she still couldn’t escape the cycle that Timberlake set in motion: she describes her 2004 Onyx Hotel tour in the aftermath as “too sexual, for a start.
</item 1>

<item 2>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
In conversation with his brother Jason, 35, the two addressed concerns from longtime football enthusiasts who were frustrated by the constant Swift references
throughout Sunday’s game at MetLife Stadium.

Not only did the football league promote last Sunday’s game by speculating whether the Grammy winner will attend the Chiefs-Jets match, but cameras were
constantly panning to the star-studded suite. During the game, Swift was joined an A-list group of guests, including her best friends Blake Lively and Ryan
Reynolds, Game of Thrones star Sophie Turner, and actor Hugh Jackman.

The NFL also introduced a special segment with Carson Daly and The Voice, where the famed host attempted to explain the specifics of football to “Swifties.”

In the New Heights podcase episode, titled “The NFL Needs to Calm Down”, Jason asked his brother plainly whether he thought the football was “overdoing it” with
the coverage of Swift during the game.

“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.
</item 2>

<item 3>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
Some found issue with these changes, but the NFL said this is a "pop cultural moment" and they are acting as such.

"We frequently change our bios and profile imagery based on what's happening in and around our games, as well as culturally," the NFL wrote in a statement. "The
Taylor Swift and Travis Kelce news has been a pop cultural moment we've leaned into in real time, as it's an intersection of sport and entertainment, and we've
seen an incredible amount of positivity around the sport."

Oct. 6: Travis and Jason think the NFL is doing too much

The NFL is really digging into the Swift-Kelce coverage, and not everyone is a fan, including the people involved.

"I think it's fun when they show who's at the game," Travis said. "I think it brings a little bit more to the atmosphere, brings a little bit more to what
you're watching. But at the same time, they're overdoing it a little bit."
</item 3>

<item 4>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"I would never have said anything like that on a previous album," she told Ryan Seacrest.

"My previous albums have always been, 'I was right, you were wrong'. What happens when you grow up is you realise that the rules in a relationship are very
blurred. It gets very complicated, very quickly."

4) Out Of The Woods

One of 1989's strengths is how the music reflects the emotional tenor of the words. Here, that means frantic anxiety.

As Taylor and her partner run from forces that seek to destroy them, they're pursued by pounding drums and rushing synths. And there's a desperation to her
voice, as though she's repeating "everything's fine, everything's fine, everything's fine," against evidence to the contrary.

By the middle 8, it seems they've made a clean getaway. Then car brakes are slammed, and it's "twenty stitches in a hospital room".

In interviews, Taylor delighted in revealing this was based on a true story.
</item 4>

<item 5>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Image source, Universal Music Group Image caption, The Bad Blood video featured an all-star "squad" of Taylor's female friends, including Cara Delevigne, Gigi
Hadid, Cindy Crawford and many more

9) Wildest Dreams

Growing up means letting go of youthful optimism, and Swift confronts that notion several times on 1989.

On one hand, she's developed an ability to laugh off her critics; on the other, she seems increasingly jaded about love.

Here, she's imagining the end of a relationship before it's truly begun. When it all goes wrong, she asks her partner, "Say you'll remember me / Standing in a
nice dress / Staring at the sunset."

"The way I used to approach relationships was very idealistic," she told Rolling Stone. "I used to go into them thinking, 'Maybe this is the one - we'll get
married and have a family, this could be forever'.
</item 5>

<item 6>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"Yeah, my personal life -- that's not so personal. I did this to myself Jason, I know this. What's real is that you know it is my personal life. ... She's not
in the media as much as I am doing this show every single week."

Sept. 27: Jerry Jones comments on the Swift effect

Swift is getting the owners talking, too.

"It is hard to move that needle dramatically on a great feeling, a great attraction in the NFL," Jones said Wednesday on 105.3FM The Fan in Dallas. "We get a
lot of eye balls. She blew it off the charts. It is amazing. It makes you smile. But it just shows the depth of her following, especially with her core
constituency. We all, I think, felt the same way when she's sitting up there with [Travis Kelce's] mom and she's going nuts when he's playing and those hearts
are popping out everywhere.
</item 6>

<item 7>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Why was I treated as though I were about to burn the place down and murder them all?

She is locked up, put on lithium, subject to relentless testing and mandatory therapy, can’t see her kids or dog, can’t bathe or dress in private or shut the
door to her room, has set bed and waking times. For blood tests, “the tech drawing my blood would be flanked by the nurse, a security guard and my assistant.
Was I a cannibal? Was I a bank robber? Was I a wild animal? Why was I treated as though I were about to burn the place down and murder them all?” She wonders if
her family is trying to kill her. Made slow and sclerotic, “I began to feel like I was being ritually tortured,” she writes, feeling estranged from her own
body. “If the idea of my being in that place was to heal, that was not the effect.”
</item 7>

<item 8>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
It’s difficult to extrapolate from the 2015 and 2020 numbers to today’s (which, as we have seen from the evidence presented here, will almost certainly not be
the whole story), but Bonta noted that the large figures are presented for impact, not as legal justification.

“The basic premise remains that their social media platforms are used by millions of children under 13. Whether it’s 30 percent, or 20 or 10 percent… any child,
it’s illegal,” he said. “If they were doing it at any time, it violated the law at that time. And we are not confident that they have changed their ways.”

An internal presentation called “2017 Teens Strategic Focus” appears to specifically target kids under 13, noting that children use tablets as early as 3 or 4,
and “Social identity is an Unmet need Ages 5-11.”
</item 8>

<item 9>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
He later issued an apology to Spears on Instagram, writing: “I’ve seen the messages, tags, comments, and concerns and I want to respond. I am deeply sorry for
the times in my life where my actions contributed to the problem, where I spoke out of turn, or did not speak up for what was right. I understand that I fell
short in these moments and in many others and benefited from a system that condones misogyny and racism.”

In June 2021, he also shared his support for Spears after she testified publicly in a bid to end her conservatorship, which was controlled by her father, Jamie
Spears. In a Los Angeles court hearing at the time, Spears alleged that she had been forced to take lithium against her will. She also claimed that the team
managing her conservatorship had prevented her from having another baby by refusing to remove her IUD.

“After what we saw today, we should all be supporting Britney at this time.
</item 9>

<item 10>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Since then, Swift has released four “Taylor’s Version” albums: Fearless (Taylor’s Version) in April 2021, Red (Taylor’s Version) in November 2021, Speak Now
(Taylor’s Version) in July 2023, and 1989 (Taylor’s Version) in October 2023.

“It’s not lost on me that the two great catalysts for this happening were two horrendous things that happened to me,” Swift said, when asked about the success
of her tour. “The first was getting cancelled within an inch of my life and sanity. The second was having my life’s work taken away from me by someone who hates
me.”

While Swift is arguably more famous now than ever before, the Grammy winner revealed that the constant attention and swarms of paparazzi won’t stop her from
living her life. There’s also constant rumours surrounding her personal life, specifically rumours about her relationship with Alwyn that her publicist, Tree
Paine, was forced to publicly deny.
</item 10>

<item 11>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has officially been named Time’s 2023 Person of the Year.

On Wednesday 6 December, the magazine announced Swift had earned the honour after an “editors’ assessment of the individual who most shaped the headlines over
the previous 12 months, for better or for worse”.

The recognition of Swift comes on the heels of a whirlwind year for the popstar, whose Eras Tour dominated ticket sales in 2023.
</item 11>

<item 12>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She suffers from postnatal depression and feels hopeless about her ability to protect her sons: “I got a little depressed once I was no longer keeping them safe
inside my body … I wanted them back inside me so the world couldn’t get at them.”

Every part of normal life had been stripped from me. I had no freedom and yet also no security

She has a keen sense of injustice regarding a perceived contract that she never signed with the public and the press. “They just kept acting like I owed it to
them to let the men who kept trying to catch me looking fat take photos of my infant sons.” Her postpartum body and shots of her without makeup were treated as
“some kind of a sin – as if gaining weight was something unkind I’d done to them personally, a betrayal. At what point did I promise to stay 17 for the rest of
my life?”
</item 12>

<item 13>
Taylor Swift has already sold over $100 million of tickets for her concert movie. Theaters nationwide brace for impact (Fortune, 2023-10-10T20:11:16+00:00)
After more than 70 years of regulating divisions between exhibition and distribution, the Paramount consent decrees were terminated in 2020 at the urging of the
Department of Justice, with a two-year sunset period that ran until last year.

“Innovation had effectively been stunted,” says Makan Delrahim, the former antitrust chief at the Justice Department who proposed ending the consent decrees.

Delrahim believes “Taylor Swift: The Eras Tour” — as a movie distributed by a theater chain, with nontraditional ticket prices — could “fuel new business models
to save the exhibitors.”

“There will be more appetite to experiment different models for theatrical distribution,” Delrahim says. “The industry needs it and, frankly, so do consumers.”

Meanwhile, “Taylor Swift: The Eras Tour” is poised to become the biggest concert film ever in about two days of release. Not accounting for inflation, 2011’s
“Justin Bieber: Never Say Never” holds that mark with $73.1 million across its entire run.
</item 13>

<item 14>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
When she returns home from the forced stint in a brutal rehab facility, she finds that her parents have thrown away her childhood things, including a binder
full of her poetry. “I felt an overwhelming sadness. I thought of the pages I’d written through tears. I never wanted to publish them or anything like that, but
they were important to me. And my family had thrown them in the trash, just like they’d thrown me away.” She resolves to start over with a new notebook. “In
that moment I made peace with my family,” she writes. “By which I mean that I realised I never wanted to see them again, and I was at peace with that.”

Britney writes that she is working to feel “more compassion than anger” towards them. “It’s not easy.”
</item 14>

<item 15>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
“This all started when Travis very adorably put me on blast on his podcast, which I thought was metal as hell,” she explained. “We started hanging out right
after that. So we actually had a significant amount of time that no one knew, which I’m grateful for, because we got to get to know each other.”

Swift also confirmed that when she famously first attended Kelce’s game at Arrowhead Stadium in September, she was already dating the NFL star.

“By the time I went to that first game, we were a couple. I think some people think that they saw our first date at that game? We would never be psychotic
enough to hard launch a first date,” she said.

The “Blank Space” singer also acknowledged that although her love life continues to make headlines with every Kansas City Chiefs game she attends, she’s focused
on being there for her partner. She specified that she isn’t trying to hide her relationship from the public eye.
</item 15>

<item 16>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
“That’s probably the moment in my life when I had the most passion for music,” she writes. “I was unknown and I had nothing to lose if I messed up.”

After that, her music barely gets a mention in A Woman Like Me. There’s a nod to her breaking records; the era-defining Toxic is dispensed with in a sentence.
The only album that she goes into any kind of detail on is 2007’s cult favourite Blackout, made at the height of her public struggles yet full of potential. The
studio was a refuge from the paparazzi hounding her outside; making a messy DIY video for the single Gimme More may be “by far the worst video I’ve ever shot in
my life”, but the scrappiness meant “more interesting people started noticing and wanting to work with me”. There’s some later love for 2016’s Glory, and she
writes proudly about her 2022 collaboration with Elton John, but says she has no plans to make music at the moment.
</item 16>

<item 17>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She accedes in order to retain access to her boys, but questions how she can perform at such a high level when she is perceived as “so sick that I couldn’t make
my own decisions”. She makes futile attempts to escape the arrangement; meanwhile her father sends her to rehab and Alcoholics Anonymous because she has been
taking over-the-counter energy supplements. The women at AA inspire her to start trying to take control over her life but she just hits more walls. Her father
won’t let her remove her IUD to start a family with her new boyfriend, later husband, Sam Ashgari (they have divorced since Britney completed the memoir). After
she objects to a new dance move suggested for her Vegas residency, she is sent to a solitary rehabilitation facility for two months, the most upsetting part of
the book.

Was I a bank robber? A wild animal? Why was I treated as though I were about to burn the place down and murder them all?
</item 17>

<item 18>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Only this time, she got to say everything she wanted.

Image source, Universal Music Group Image caption, The video for Shake It Off was filmed under intense secrecy

6) Shake It Off

Rattling like a bag of spanners, Shake It Off is a 21st Century Hey Mickey, a honking great party record with a chorus that drills so far into your brain you
start seeing lights.

It was the last song recorded for 1989, and Taylor arrived at the studio with a challenge for her co-writers Max Martin and Johan "Shellback" Schuster: "We need
drums that make even the person who's having a terrible night at the wedding [go], 'Oh, it's my song!'"

Like Blank Space, Shake It Off is a response to celebrity takedown culture. Taylor lists all of her supposed flaws ("She's an airhead!' 'She dates too many
men!") with a defiant "so what?
</item 18>

<item 19>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Spears is also the mother of two children – Sean, 18, and Jayden, 17 – who she shares with her second ex-husband, Kevin Federline.

According to a press release for The Woman in Me, Spears’ memoir is “a brave and astonishingly moving story about freedom, fame, motherhood, survival, faith,
and hope”. Spears’ book is also expected to cover the high-profile legal battle to end her conservatorship, which was dissolved in November 2021 after 13 years.

Throughout her time in the spotlight, specifically during the ‘90s and 2000s, Spears has been in some heavily publicised romances. Unsurprisingly, fans are
anticipating that Spears will detail some of her relationships in The Woman in Me, which is set for release on 24 October.

From Justin Timberlake to her now ex-husband Sam Asghari, here’s everything we know about Britney Spears’ relationships and rumoured romances.
</item 19>

<item 20>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Elsewhere in the conversation with Time, the “Anti-Hero” singer said that she tries not to let the paparazzi get to her, even though so many of her outings with
pals go on to make headlines. “Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter. Yes, if I go out
to dinner, there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends,” she said.

She emphasised how her perspective about the media has changed over the years, adding: “Life is short. Have adventures. Me locking myself away in my house for a
lot of years - I’ll never get that time back. I’m more trusting now than I was six years ago.”

With his love life in the public eye, Kelce has also opened up about his relationship with Swift.
</item 20>

<item 21>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
This isn't fun for me'. But my second reaction was, 'Hey, that's actually a really interesting character they're writing about: She jet-sets around the world
collecting men - and she can get any of them, but she's so clingy that they leave and she cries in her marble bathtub surrounded by pearls'.

"I was like, 'I can use this'."

Lyrically, it contains some of her best lines ("darling, I'm a nightmare dressed like a daydream") and there's a delicious attention to detail - with a pen
click signifying every new entry in her little black book.

Shout-out to all the lonely Starbucks lovers; and to Chicago indie band Hard Femme who took that misheard lyric and made it a real song. Those search engine
clicks gotta be lucrative, right?
</item 21>

<item 22>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's so weird to look back on that - you've got heaping piles of discarded songs that I really loved when I wrote them and I still love now.

"I just need each album to have its own sound and if it doesn't fit that sound, it doesn't matter if it's a great song to me."
</item 22>

<item 23>
Is the Coverage on Taylor Swift and Travis Kelce’s Romance Nothing but Hype or a Pivotal Shift in NFL’s Century-Old Game Plan Towards Global Recognition?
(Essentially Sports, 2023-10-25T21:30:41+00:00)
$28 billion. And just like that, their target will be met.

USA Today via Reuters NFL, American Football – Taylor Swift reacts while sitting next to Donna Kelce watching the Kansas City Chiefs vs Chicago Bears game
during the first half at GEHA Field at Arrowhead Stadium, Kansas City, Missouri, USA | Image Credits: Denny Medley-USA TODAY Sports/Reuters

And while analysts like Stephen A. Smith may go around telling people they should be embarrassed of themselves for getting so invested in someone else’s
relationship, it is good business for the NFL. TMZ reported that #87’s jersey sales skyrocketed 400% and his Instagram followers by 276%. Yahoo Finance reported
a 53% rise in viewership among teen girls. Who would have thought in their wildest dreams that the NFL would manage to rope in this part of the population? The
October 1 matchup between the Chiefs and the Jets drew 27 million viewers.
</item 23>

<item 24>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
There is no love lost for her family

Britney writes about how she was always made to feel she was not enough as a kid; her father’s alleged alcoholism, neglect and brutal standards; how her mother
always made her feel less-than. She is unsparing about her younger sister, who she characterises as a brat who was spared the pain of Britney’s impoverished
childhood because of her success. “My mom and I had to witness the ugliness and the violence without believing that there was anywhere else to go.”

Even once Britney is the biggest pop star in the world, her parents still seem to have an undue say in the running of her career, including making her sit for
humiliating TV interviews – why exactly that is remains unanswered. Her father seizes control legally with the imposition of the conservatorship, telling her,
chillingly, “I am Britney Spears now,” and her family appear to live off her dime while she lives an extraordinarily restricted life.
</item 24>

<item 25>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Timberlake, she likens herself to Benjamin Button. “Somehow that year, in becoming more vulnerable I started to feel like a child again.”

The situation worsens amid postnatal depression. “It was as if some part of me became the baby,” she writes in an apologetic section about transferring her
frustrations on to her interior decorator. “One part of me was a very demanding grown woman yelling about white marble, while another part of me was suddenly
very childlike.”

When she is placed under the conservatorship, she not only regresses, but feels dehumanised by the constant scrutiny. “I became a robot. But not just a robot –
a sort of child-robot. The conservatorship stripped me of my womanhood, made me into a child.”
</item 25>

<item 26>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“I think some people think that they saw our first date at that game? We would never be psychotic enough to hard launch a first date,” Swift joked.

Back in April, it was revealed that Swift and British actor Joe Alwyn had broken up after six years of dating. The former couple were notoriously private, as
the Conversations with Friends star previously expressed he didn’t want to discuss their relationship because it “feeds into a weird part of the culture”.

Now, Swift has shared that she has nothing to hide in her new relationship with Kelce. “When you say a relationship is public, that means I’m going to see him
do what he loves, we’re showing up for each other, other people are there and we don’t care,” she said. “The opposite of that is you have to go to an extreme
amount of effort to make sure no one knows that you’re seeing someone. And we’re just proud of each other.”
</item 26>

<item 27>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Like, through thick and thin, sickness and health love," she told MTV.

Drawing directly on conversations she'd had with Lena, the song blends scenes of everyday life (burnt toast, borrowed shirts) with moments of incandescent
tenderness (kisses on sidewalks and late night heart-to-hearts).

In an interview with Elle, Taylor talked wistfully about the couple's ease around each other.

"I've never had that," she said. "[But] now I have more of a grasp on the fact that when you're in a state of infatuation and you think everything that person
does is perfect, it then - if you're lucky - morphs into a real relationship when you see that that person is not in fact perfect, but you still want to see
them every day."
</item 27>

<item 28>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
For example, will the survey go to under 13 year olds? Since everyone needs to be at least 13 years old before they create an account, we want to be careful
about sharing findings that come back and point to under 13 year olds being bullied on the platform.”

In 2021, another, studying “child-adult sexual-related content/behavior/interactions” (!) said she was “not includ[ing] younger kids (10-12 yos) in this
research” even though there “are definitely kids this age on IG,” because she was “concerned about risks of disclosure since they aren’t supposed to be on IG at
all.”

Also in 2021, Meta instructed a third-party research company conducting a survey of preteens to remove any information indicating a survey subject was on
Instagram, so the “company won’t be made aware of under 13.”
</item 28>

<item 29>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
- Enchanted

Mahomes said he had not met Swift when asked in the postgame press conference, but he did meet her at the afterparty.

"Yea I met her. She's really cool. Good people," Mahomes said, via KSHB 41 in Kansas City. "But like Trav said, imma let them have their privacy and just keep
it moving."

Sept. 27: Kelce breaks his silence

"And they said speak now." -Speak Now

Kelce addressed Swift's attendance on his podcast, saying, "To see the slow-motion chest bumps, to see the high-fives with Mom, to see how Chiefs Kingdom was
all excited that she was there -- that s--- was absolutely hysterical. It's definitely a game I'll remember, that's for damn sure."

He is not going to speak too much more about the relationship, noting that he wants "to respect both of our lives."

"Yeah, my personal life -- that's not so personal.
</item 29>

<item 30>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Justin Timberlake, she dreams about quitting pop and opening a shop: “With the gift of hindsight I can see that I hadn’t given myself
enough time to heal from the breakup,” she writes – though her requests for a break are not only turned down, but interpreted as a potential sign of sedition to
be quashed and interrogated.

In this restrictive environment, small kindnesses take on an outsized significance, like Paris Hilton encouraging her “to have fun for the first time in a long
time”, leading to some infamous nights out with the heiress and Lindsay Lohan. Once again, Britney is seen as stepping out of bounds, potentially damaging her
earning potential. One night she has her mother babysit her sons; when she returns home tipsy, Lynne Spears screams at her, she writes. “The shame I felt killed
my heart. I stood there, reeling, and thought, OK, I guess it’s forbidden for me to party.”
</item 30>

<item 31>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“It saddens me to hear that my ex husband has decided to discuss the relationship between me and my children … As we all know, raising teenage boys is never
easy for anyone,” she wrote. “It concerns me the fact that the reason is based on my Instagram … It was LONG before Instagram … I gave them everything … Only
one word: HURTFUL.”

Asghari went on to accuse Federline of using their sons to “vilify” his wife, and claimed her ex-husband’s “gravy train will end soon, which probably explains
the timing of these hurtful statements.”

“There is no validity to his statement regarding the kids distancing themselves and it is irresponsible to make that statement publicly,” Asghari wrote on
Instagram.

Following Federline’s comments, Spears’s attorney Mathew Rosengart said in a statement obtained by Variety that Federline has created “legal issues” for
himself, after posting private videos of Spears sternly speaking with her children without her consent.
</item 31>

<item 32>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
It’s posting selfies – either nude or clothed – on Instagram. Some people don’t understand the latter, she writes, “but I think if they’d been photographed by
other people thousands of times, prodded and posed for other people’s approval, they’d understand that I get a lot of joy from posing the way I feel sexy and
taking my own picture, doing whatever I want with it.”

Her ‘virginity’ was a prison

Britney was actively marketed as a virgin, a repulsive, archaic premise that made the 16-year-old’s inevitable sexuality into a kind of timebomb. In fact, she
writes happily, she first slept with her older brother’s best friend when she was 14. Yet the fixation on her “purity” took “the focus off me as a musician and
a performer”, she writes. “All some reporters could think of to ask me was whether or not my breasts were real (they were, actually) and whether or not my hymen
was intact.”
</item 32>

<item 33>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Taylor Swift has revealed when her relationship with Travis Kelce really started.

The 33-year-old singer spoke candidly about the romance during a recent interview with Time after she was named the magazine’s 2023 Person of the Year. Speaking
to the publication, she said she connected with Kelce in July, after he confessed on his podcast that he tried and failed to give her a friendship bracelet with
his phone number on it during one of her Eras Tour shows.
</item 33>

<item 34>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
At the time, her representative also confirmed to People: “Yes, she’s single.”

The news of the breakup came asTMZ alleged that Lucado had cheated on Spears, with claims that her father had purchased a video of the infidelity incident. The
publication later alleged that Lucado denied the cheating allegations, with claims that he was seeing the other woman while he and Spears were broken up.

Despite their seemingly rocky history, Lucado showed his support for his ex in a 2016 interview with theNew York Times about her conservatorship. Speaking to
the publication, he said: “If anyone knew the real Britney, they would know that she would rather be remembered for being the great mother she is rather than
the artist she is.”

The publication also noted that Lucado denied the infidelity claims from Spears’ father, before he added: “And if anyone could see her interactions with her
kids, they would know that there is no need for a conservatorship over Britney’s personal life.”
</item 34>

<item 35>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“I already read it and I’m very proud of her,” he told the outlet. “She put a lot of work into it, and it was very hard. I’m definitely excited, and I’m
extremely proud of her. It was a very tough one [to write].”

The Woman in Me is due for release on 24 October in the US and UK.
</item 35>

<item 36>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“I’m her first husband, I’m here to crash the wedding.”

After he was arrested for trespassing, vandalism, and battery, a security guard later revealed in court that when Alexander crashed the wedding, he also tried
to enter Spears’ bedroom.

Kevin Federline: 2004 to 2007

(Getty Images)

Kevin Federline and the “Gimme More” singer first met at a club in 2004, before they went on to start their relationship. After just three months of dating, the
two tied the knot in September of that year during a surprise wedding ceremony. Weeks before their one-year wedding anniversary, they welcomed their first
child, son Sean Preston.

In 2006, they welcomed their second son, Jayden James. At the time, Federline was already a father to his daughter Kori Madison, now 20, and son Kaleb Michael,
18, with ex-fiancée Shar Jackson.
</item 36>

<item 37>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
"'Taylor wants a guy that's into his career, does his own thing, but is also supportive and understands her,' the source added, "and Travis fits those
qualities.'"

Sept. 25: Kelce jersey sales skyrocket

Call it what you want; I'll call it the Taylor Swift effect. Since Swift attended the Chiefs game, there was a 400% increase in Kelce jersey sales, his podcast
is No. 1 overall on Apple, he gained over 380,000 Instagram followers, and 24.3 million people watched the game, making it the No. 1 game last week, according
to Front Office Sports.

There has also been triple the "Chiefs" searches on web and triple the Chiefs ticket sales on Stubhub.

Sept. 25: Belichick weighs in on the matter



New England Patriots head coach does not always comment on situations that take place in the game, but he was fully ready with a sound bite on the Kelce-Swift
situation.
</item 37>

<item 38>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
That same day, the fitness instructor officially filed for divorce, listing the date of separation as 28 July 2023 and citing "irreconcilable differences" as
the reason for the split.

In a statement posted online, Asghari said: “After six years of love and commitment to each other my wife and I have decided to end our journey together. We
will hold onto the love and respect we have for each other and I wish her the best always. S*** happens. Asking for privacy seems ridiculous so I will just ask
for everyone including the media to be kind and thoughtful.”

Days before Spears’ memoir release date, Asghari revealed that he had obtained an early copy of the book. In an interview with TMZ on 13 October, he explained
that he wasn’t worried about anything being written in the book that could paint him in a negative light.

“I already read it and I’m very proud of her,” he told the outlet.
</item 38>

<item 39>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Charlie Ebersol: 2014 to 2015

Spears and Charlie Ebersol, who is known as the co-founder of Tempus Ex Machina and The Company, started dating in 2014. While their relationship was fairly low
profile, Spears previously opened up about why she was drawn to Ebersol in the first place.

“There was something about him that made me feel really peaceful and at ease. And he was really hot, too. Adorably hot,” she told People at the time.

They officially broke up in June 2015 after eight months of dating.

Sam Asghari: 2016 to 2023

(Getty Images)

The pair first met in 2016 when Asghari was cast as Spears’ love interest in the music video for her song, “Slumber Party”, from her 2016 album Glory.
</item 39>

<item 40>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Spears has previously been open about her co-parenting struggles with Federline. Last year, he claimed their children were keeping their distance from Spears,
who was embracing her newfound freedom following the end of her 13-year long conservatorship by posting several nude images on social media.

From marriage to divorce to co-parenting, here is a complete timeline of the drama between Britney Spears and Kevin Federline.

The “Toxic” singer and the former backup dancer began their whirlwind romance back in 2004, when Federline worked as a backup dancer for Justin Timberlake.
Federline recalled his first meeting with Spears to People back in 2008.

“I met her at a club in Hollywood, Joseph’s. Our eyes met and that was it. We just hit it off right away,” he said. “I learned real fast how much of a whirlwind
the press and everything was.”
</item 40>

<item 41>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
But that's for another article..

17-21) Tracks from the vault

In 2014, Taylor said she had recorded more than 100 songs for 1989 - so the relative scarcity of vault tracks is intriguing.

Whereas previous re-recordings have featured full discs of bonus material, this album gets just five new additions: Slut!, Say Don't Go, Now That We Don't Talk,
Suburban Legends and Is It Over Now?

As a whole, they feel like dry runs for the songs that made the cut. The melodies aren't as crisp, the lyrics aren't as sharp. Suburban Legends has a
particularly clunky line about letting a partner's indiscretions slide "like a hose on a slippery plastic summer".

Slut! has piqued fans' interest for its title alone - who might she be talking about? No-one, it turns out.
</item 41>

<item 42>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
And we’re just proud of each other.”

Speaking to Time, Swift detailed exactly how much work went into preparing for her Eras Tour - where each concert spans over 180 minutes, with more than 40
songs from at least nine albums and 16 costume changes. “Every day I would run on the treadmill, singing the entire set list out loud,” she told the outlet.
“Fast for fast songs, and a jog or a fast walk for slow songs.”

She also stopped drinking and underwent three months of dance training, something that she admitted is “not my strong suit”. After she plays a run of shows,
Swift said she takes a day to rest and recover before taking the stage once again. “I do not leave my bed except to get food and take it back to my bed and eat
it there,” she revealed. “It’s a dream scenario.”

Most recently, a heat wave in Brazil forced Swift to postpone the second night of her Eras Tour shows in Rio de Janeiro.
</item 42>

<item 43>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
“I think everybody is just, like, overwhelmed,” the Chiefs tight end replied.

“Take away your feelings for Taylor,” Jason added. “What is your honest opinion on how the NFL is treating celebrities at the games?”

Seemingly beating around the bush, Kelce remarked: “I think it brings a little more to the atmosphere, it brings a little bit more to what you’re watching. But
at the same time, I think...”

Before his younger brother could continue his thought, the Philadelphia Eagles player chimed in: “They’re overdoing it.”

“They’re overdoing it a little bit, for sure,” Kelce agreed. “Especially my situation. I think they’re just trying to have fun with it.”

While sports lovers may not be too pleased with the NFL’s focus on Swift, the football league’s supposed obsession with ongoing Swift and Kelce dating rumours
has paid off in terms of ratings.
</item 43>

<item 44>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
In the couple’s initial divorce settlement, which was finalised in July 2007, they were granted joint legal and physical custody over their two children.

Singer Britney Spears with ex-husband Kevin Federline in 2006 (Getty Images)

Spears suffered a mental breakdown in 2008 and was placed on a 5150 psychiatric hold, leaving Federline to retain sole custody of their boys. In February of
that year, Spears entered into a conservatorship with her father Jamie Spears, who took complete control over her assets.

Since Spears and Federline’s divorce, it seemed the two were amicably co-parenting their sons when they were seen supporting their boys at their soccer and
football games. Federline moved on to his now wife Victoria Prince. The two share daughters Jordan Kay, 12, and Peyton Marie, nine.

In 2018, things took a turn when Federline requested an increase of $20,000 a month more child support payments from Spears.
</item 44>

<item 45>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Jason Trawick: 2009 to 2013

(Getty Images)

Spears and Jason Trawick, a Hollywood agent who worked at WME, started dating in 2009 before getting engaged in December 2011. However, they called off their
engagement in January 2013 and announced they were going their separate ways.

"Jason and I have decided to call off our engagement. I’ll always adore him and we will remain great friends," Spears said in a statement sent by her
representative to ABC News.

Trawick also issued a statement at the time, saying: “As this chapter ends for us a new one begins. I love and cherish her and her boys and we will be close
forever."

David Lucado: 2013 to 2014

Two months after ending her engagement with Trawick, Spears started dating David Lucado, who worked at a legal firm. In August 2014, Spears announced their
split, tweeting: “Ahhhh the single life!”
</item 45>

<item 46>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
All proceeds from the track are donated to New York City's Public Schools.

Image source, Getty Images Image caption, The original version of 1989 won three Grammy Awards, including Album of the Year

2) Blank Space

Pretty much the perfect Taylor Swift song, Blank Space takes every horrible rumour that was being spread about her love life, and amplifies it with satirical
relish.

"In the last couple of years the media have had a wonderful fixation on painting me as the psycho serial dater girl," she told the Grammy Museum. "Every article
was like, 'Here's Taylor Swift standing near some guy. WATCH OUT GUY!'

"My first reaction was to be like, 'That's a bummer. This isn't fun for me'.
</item 46>

<item 47>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
They now share two daughters, Jordan, 10, and Peyton, eight.

Adnan Ghalib: 2007 to 2008

Spears met British paparazzo Adnan Ghalib in 2007, shortly after she decided to shave her hair off in front of approximately 70 photographers. They only dated
for about a year, as the relationship ended in 2008.

In the 2021 Netflix documentary, Britney vs Spears, Ghalib spoke candidly about the romance, claiming: “It was a very simple, very easy relationship, you know?
But between her fans and everybody else, it was like I was the bad guy.”

The former couple split in 2008 after Ghalib reportedly pulled a pregnancy test-buying stunt for publicity. In 2009, Jamie Spears filed a restraining order
against Ghalib, which ordered the photographer to stay away from the singer and her family until 2012.
</item 47>

<item 48>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
It's certainly not her finest work - a vendetta disguised as a pop song, with a ham-fisted chorus that lacks her usual melodic instincts.

Swift hinted in a Rolling Stone interview that the track was about a high-profile female musician, saying the person did something "so horrible" that made them
"straight-up enemies."

Although she tried not to identify the target, it became common knowledge that Katy Perry was in the crosshairs, after allegedly "stealing" Taylor's backing
dancers just before a tour.

Perry confirmed the situation on James Corden's Carpool Karaoke, adding that she "tried to talk to [Swift] about it and she wouldn't speak to me."

The pop divas eventually reconciled in the 2019 video for You Need To Calm Down... but Taylor's still performing Bad Blood on her Eras tour.
</item 48>

<item 49>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"A farewell to twang," the New York Times called it.

Taylor knew change was necessary. "I don't have the option of making music that sounds just like what I've done before," she told Popcrush. "People will see
right through it. They'll see that I was lazy".

Her label begged her to add three country-leaning songs to the album, but she stood her ground.

"At a certain point, if you chase two rabbits, you lose them both."

Welcome To New York itself is the sound of creative and personal freedom. Taylor moved to the city in 2014 and was amazed to discover she could go grocery
shopping largely incognito.

"My friends and people who know me have commented that I'm physically different since being here," she told NYC radio station WKTU. "That song is about the
endless hopefulness and endless possibilities that New York seems to present," she added in an interview with Access Hollywood.

All proceeds from the track are donated to New York City's Public Schools.
</item 49>

<item 50>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
When she arrives, so does another Swat team. “I’m a five-foot four-inch pop singer who calls everyone sir and ma’am,” she writes. “They treated me like I was a
criminal or predator.”

The conservatorship clamps down on her life, ruled by her father, who she alleges was “an alcoholic, someone who’d declared bankruptcy, who’d failed in
business, who’d terrified me as a little girl”. She is medicated and surveilled. Her dates are required to agree to background checks and blood tests, made to
sign NDAs, and receive a full rundown of her sexual history before the first date. “The insanity of this system kept me from finding basic companionship, having
a fun night out or making new friends – let alone falling in love,” Britney writes. “It was just the worst thing that could possibly ever happen to my music, my
career and my sanity.”
</item 50>

<item 51>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
As the restrictions get tighter, Britney acts out – and ends up legally denied her freedom for 13 years, under a conservatorship controlled by her father.

Her body felt like public property

From day one, Britney’s body is made fair game. Between the videos for her debut single, Baby, One More Time and Sometimes, the media noted her changing
physique and concluded that the 16-year-old must have had breast implants – rather than simply going through puberty. She notices that talkshow hosts ask her
about her chest, while her boyfriend Justin Timberlake gets asked serious questions about his music. It’s one of many pressures that ultimately push her to
breaking point, and when she writes about shaving her head in a salon in 2007, she frames it as a “fuck you” to the impossible beauty standards she felt forced
to live up to: “I’d smiled politely while TV show hosts leered at my breasts, while American parents said I was destroying their children by wearing a crop
top.”
</item 51>

<item 52>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
While celebrities like Madonna, Paris Hilton, and Selena Gomez attended the festivities, Spears’ close family – her mother Lynne, father Jamie, sister
Jamie-Lynn, and her two sons – were not present at the nuptials.

Although their marriage last only 14 months, with Asghari filing for divorce in August this year, the actor and former dancer was previously praised by fans
over his continuing support for Spears amid her conservatorship and custody battle.

In August 2022, Federline claimed during an interview with ITV that their teenage sons found it “tough” seeing their mother sharing nude photos online, saying:
“I can’t imagine how it feels to be a teenager having to go to high school.”

Spears hit back at her ex-husband in a statement shared to her Instagram Stories responding to Federline’s “hurtful” comments.
</item 52>

<item 53>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
This was the first, albeit tentative, confirmation from the media rather than from amateur Swiftie sleuths on social media. So, it was only a matter of time
before it entered the cultural zeitgeist. Blank Space Which it did, very quickly. On September 17, Kelce scored a touchdown against the Jaguars, triggering CBS
announcer Ian Eagle to reference Swift’s 2014 hit, Blank Space.

“Kelce finds a blank space for the score,” Eagle said, poking fun at Kelce’s newfound Swiftie status. And Eagle wasn’t the only one, with NFL Network’s Rich
Eisen including a slew of lyric references during a preshow, including “shake it off”, “anti-hero” and “all too well”. Kelce even acknowledged the
tongue-in-cheek banter, commenting on a montage video of the preshow that Eisen posted to Instagram. “Well played Rich … Well played,” he wrote. Suddenly,
football fans and Swifties were living through this surprising collision.
</item 53>

<item 54>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
The singer made a powerful testimony at her conservatorship hearing that June, where she spoke out for the first time about her experience, calling for those
involved to be “put in jail”. Following the testimony, Federline expressed his support for the singer, saying he “only hopes the best for her because when the
best for her is achieved, it’s the best for their kids.”

“The best thing would be for their mom to be healthy and happy,” his lawyer said in a statement at the time. “And if either of those things aren’t true, it
doesn’t provide for the best setting for custody to be exercised.”

Britney Spears poses with her sons Jayden James and Preston at Dodger Stadium in 2013 (Getty Images)

In June 2022, Spears married her longtime boyfriend Sam Asghari during an intimate ceremony at their Los Angeles home, just one month after the singer revealed
she suffered a miscarriage.
</item 54>

<item 55>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Justin Timberlake: 1998 to 2002

(Getty Images)

Spears’ first highly public relationship was with her co-star in the Mickey Mouse Club, Justin Timberlake. The pair made many appearances together while dating,
including the iconic moment Spears and Timberlake wore matching, all-denim outfits to the American Music Awards in 2001.

They broke up in 2002, shortly after Timberlake and his boy band NSYNC released their hit tune, “Cry Me a River”. At the time, the music video for the song –
which featured a woman with blonde hair - sparked rumours that he and Spears broke up because she allegedly cheated on him.

Although the former couple never acknowledged the breakup rumours, the 2021 Framing Britney documentary addressed some of the public scrutiny that Spears faced
during the 2000s. Following the documentary’s release, Timberlake came under criticism for the way he acted following his breakup with Spears.
</item 55>

<item 56>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Image source, Republic Records Image caption, Artwork for the re-recorded version of 1989 has a beach vibe, in contrast to the Polaroid aesthetics of the
original

16) New Romantics

Proof that even Taylor gets it wrong: New Romantics, one of her most beloved songs, was originally held back as a bonus track.

It's a beast, though. An armour-plated anthem to resilience and power, where Taylor takes all the pain that's come her way and uses it to make herself
bulletproof.

"Heartbreak is the national anthem, we sing it proudly," she declares, while glitterballs shatter around her. "I could build a castle out of all the bricks they
threw at me."

That defiance also acts as a bridge to her next album, Reputation, which dismantled her critics, one-by-one.
</item 56>

<item 57>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Per a Deadline report, NBC accumulated an average of 27m viewers, with a total of 29m tuning in during the second quarter.

“That’s gotta be an NFL record right there,” Jason told his younger brother, as he noted that about two million viewers were estimated to be female. “I think a
few of them might have been there for Deadpool and, uh, Taylor.”

In the YouTube comments under the New Heights video podcast, fellow fans shared their opinions on the NFL’s coverage. “I agree the NFL needs to calm down.
Pissing off football fans is not how you repay your boi [sic] for all he’s done for the sport,” one person wrote.

Another remarked: “Pity Taylor. All she ever wanted was to watch the football game. Now she’s being blamed and hated left and right.”

“The NFL needs to seriously tone it down and let Trav and Tay explore this beautiful thing they have going on in peace,” someone else said.
</item 57>

<item 58>
Travis Kelce faces backlash after comments about ‘finding a breeder’ resurface amid Taylor Swift romance (The Independent - Life and Style,
2023-10-30T20:56:51+00:00)
“Forever grateful that Joe Alwyn kept to himself and is private,” one wrote, referring to Swift’s ex-boyfriend of six years.“It’s what makes Travis Kelce’s
statements from earlier this year about ‘finding a breeder’ cringe. You are free to say what you want but are not free from the consequences of doing so. Women
are not objects.”

“‘Start breeding,’ oh this is just f***ing disgusting and it’s setting us back so much,” another added. “Women aren’t baby-making machines, women are more than
the man they’re dating, we are more than marriage.”

“Ain’t a Swiftie but don’t have to be one to be weirded out about Travis Kelce calling women ‘breeders,’” someone else wrote. “That ain’t it dude.”

Many other people went on to criticise claims about Swift and Travis having children soon, since the singer is about to return to her Eras Tour until November
2024.
</item 58>

<item 59>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
In interviews, Taylor delighted in revealing this was based on a true story.

"It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something
really major and traumatic happen to me and they don't know about it," she told NPR.

In her Rolling Stone cover story, Taylor said the crash actually occurred on a snowmobile "with an ex who lost control and wrecked it so badly that she saw her
life flash before her eyes".

That prompted speculation the track was also about Styles: The couple had gone skiing with Justin Bieber and Selena Gomez in late 2012.

The lyrics also reference their identical aeroplane necklaces and a December break-up, which ties in with the "Sad Taylor on a boat" meme.

But the singer steadfastly refuses to name names: "The fact that I've never confirmed who those songs are about makes me feel like there is still one card I'm
holding," she told GQ.
</item 59>

<item 60>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
This is the biggest and riskiest part of the project. While earlier remakes were largely made with a live band, 1989's pop landscapes are full of squelchy,
processed synths and treated vocals.

Fans can now find out how it has turned out - and hear five new tracks from the star's vault, expanding on the themes and relationships she explored on the
original.

Here's a guide to all the songs and what Taylor has said about them.

1) Welcome To New York

"It's a new soundtrack, I could dance to this beat forever more."

We open with a mission statement: The old Taylor can't come to the phone right now, she's dead.

After building her reputation in Nashville, 1989 jettisons the banjos in favour of insistent, needling beats and tales of bohemian nightlife. "A farewell to
twang," the New York Times called it.

Taylor knew change was necessary.
</item 60>

<item 61>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?

In August, the brothers discussed Kelce’s moustache, which he grows every year in honour of the Chief’s head coach, Andy Reid. When asked if he knew what Swift
thought of the ’stache, he remained cheekily confident. “We’re not going to bring up Taylor Swift in this episode,” he says, “But something tells me she’s going
to like it.” There was a clear suggestion Kelce had made contact with Swift, and was even going to meet her. Shortly after that remark, Kelce took a week off
training to visit New York City, according to Vulture. Coincidentally, this happened to be while Swift was re-recording her 1989 tracks in the city. Welcome to
New York

The suspicion the two were actually spending time together in New York was seemingly confirmed in September when it was reported they had been “quietly hanging
out” for months.
</item 61>

<item 62>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"I just wrote it down, and closed the book and put it back on my nightstand, like, 'Oh, that was fun'.

"Then all of a sudden in my head, I started hearing this melody - and that's when I realised it was going to be a song."

12) I Know Places

Continuing the theme of thwarted love, I Know Places was fuelled by Taylor's increasing anxiety over holding down relationships in the spotlight.

"I was like, 'No-one's gonna sign up for this. There are just too many cameras pointed at me, there are too many ridiculous elaborations on my life'," she told
the Grammy Museum.

"But I decided to write a love song [about] what would I say if I met someone really awesome and they were like, 'Hey, I'm worried about all this attention you
get'.

"So I wrote this song about, 'Hey, I know all these places we could hide. We can outrun them'."
</item 62>

<item 63>
Taylor Swift has already sold over $100 million of tickets for her concert movie. Theaters nationwide brace for impact (Fortune, 2023-10-10T20:11:16+00:00)
Swift’s camp was motivated to get the film out even as her stadium tour continues internationally. The tour, which is projected by Pollstar to gross some $1.4
billion, crashed Ticketmaster’s site, saw sky-high resale mark-ups and left many fans priced out.

The movie, directed by Sam Wrench, would be a way for millions more to experience the Eras Tour. Adult tickets are being sold for $19.89, a reference to her
birth year and 2014 album, a re-recording of which is due out Oct. 27. That’s higher than the average movie ticket but several thousand less than many tickets
to see Swift live.

It’s arriving uncommonly fast, too, just a little over two months since the SoFi shows. Speed was one reason Swift’s father, Scott Swift, is said to have sought
out a direct deal with AMC. Swift produced the film, herself, and, with 274 million followers on Instagram, didn’t need a studio to promote it.
</item 63>

<item 64>
Subtle as a sledgehammer, a pop icon returns (The Age, 2023-11-30T18:30:00+00:00)
That these songs are so interesting makes the album’s overemphasis on understated ballads frustrating. Songs like opener Cut My Hair, Grave and Stay Done are
slick but dour affairs, the classic Tedder formula. Messier, which weirdly is not another Canadian nod to ice hockey great Mark, shows McRae has the range, a
bitter ballad where she allows her vocals to get raw and unhinged.

But the real pleasure of Think Later lies in McRae’s idiosyncrasies and their potential; she’s a throwback in the classic singing-dancing Britney mould. With a
pop landscape led by serious singer-songwriters like Taylor Swift, SZA, Rodrigo and Eilish; Dua “go girl, give us nothing” Lipa; the fame-trolling Doja Cat; and
the introverted to the point of obscurity PinkPantheress, there’s a vacancy for McRae’s particular set of popstar skills. If she’s greedy enough, the platform
feels like it’s hers to grab.
</item 64>

<item 65>
‘Mass breach of privacy’: TikTok under fire for tracking users online (The Age, 2023-12-25T18:00:00+00:00)
Strategic Policy Institute researcher Samantha Hoffman said the data collected by TikTok’s pixel was similar to that of US-based tech giants Google and Meta,
but the difference was “the intent”.

Advertising data had “incredible propaganda value”, she said. Loading “If you think about that, plus the access that TikTok is required to give the Chinese
government, that’s the problem.” In November 2022, TikTok changed its privacy policy to make it explicitly clear user data can be accessed by some employees
from across the world, including China. “They talk about how even data collected overseas can be used by the company and its partners, and would be kept private
unless security organisations make demands of it,” Hoffman said.
</item 65>

<item 66>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Spears went on to file for divorce from Federline in November 2006, citing “irreconcilable differences” as the reason for their split. In the couple’s initial
divorce settlement, they were granted joint legal and physical custody over their two children. While it appeared they were on good terms after their
separation, things took a turn in 2018 when Federline requested an increase of $20,000 a month more child support payments from Spears.

After months of court discussions, Federline rejected Spears’s child support proposal and instead doubled his request to $40k a month. A settlement was finally
reached between the two parties in September 2018, with Spears increasing her child support payments to Federline from $20,000 to $35,000 a month.

Federline has since moved on from the relationship, as he married volleyball player Victoria Prince in 2013. They now share two daughters, Jordan, 10, and
Peyton, eight.
</item 66>

<item 67>
Travis Kelce says he tried to delete his ‘nonsense’ tweets before they went viral (The Independent - Life and Style, 2023-11-25T14:50:53+00:00)
In an interview withWSJ Magazine published on 20 November, Kelce described how much he admires Swift’s skills as a musician. “I’ve never been a man of words.
Being around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

Although he acknowledged the public attention his personal life has received because he’s “never dated anyone with that kind of aura around them,” Kelce
specified that he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life.

“When she acts like that I better not be the one acting all strange.”
</item 67>

<item 68>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
That brief moment of freedom doesn’t last long: a year later, she is ambushed into the conservatorship, which also exercises jurisdiction over her body. “No
matter how much I dieted and exercised, my father was always telling me I was fat,” she writes. Punishing gym routines leave her feeling “out of my mind”. Her
diet is monitored – along with every other detail of her life – while she goes through the motions of her Las Vegas residency. “My body was strong enough to
carry two children and agile enough to execute every choreographed move perfectly onstage. And now here I was, having every calorie recorded so people could
continue to get rich off my body.”

Here I was, having every calorie recorded so people could continue to get rich off my body

On the other side of the conservatorship, she writes, freedom looks like being able to gain weight without anyone “shouting at me”. It’s eating chocolate again.
It’s posting selfies – either nude or clothed – on Instagram.
</item 68>

<item 69>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
87 with gusto.

The 12-time Grammy winner appeared to shout “let’s f—ing go” after Kelce scored a touchdown before chest-bumping other Chiefs fans. Considering Swift is known
for being an Eagles fan – she’s originally from Pennsylvania and even references the team in Gold Rush – her outward support for the Kelce’s team seemed extra
telling. Swifties were immediately on high alert, with some capturing Swift and Kelce leaving the stadium together. If that wasn’t enough, extra observant fans
were able to note that Kelce’s blue and white post-game outfit was supposedly labelled the “1989 bedroom painting set”, perhaps an intentional reference to
Swift’s 2014 album, which she will be re-releasing in October. Swift’s penchant for Easter eggs must be rubbing off on the tight end player. As if this weren’t
enough subtle references already, they were also spotted in a “getaway car” together outside the stadium – potentially a nod towards her 2017 track.
</item 69>

<item 70>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
At what point did I promise to stay 17 for the rest of my life?”

What she knows now, she writes, “is that every part of normal life had been stripped from me – going out in public without becoming a headline, making normal
mistakes as a new mother of two babies, feeling like I could trust the people around me. I had no freedom and yet also no security. At the time I was also
suffering, I now know, from severe postpartum depression.” It left her suicidal. “I’ll admit it, I felt that I couldn’t live if things didn’t get better.”

She knows she is trapped in a state of arrested development

It’s often said that famous people are frozen in time at the age they became famous, but Britney was never allowed to be a real child, nor a real adult. Her
2001 single I’m Not a Girl, Not Yet a Woman may be her most prescient recording; she is painfully conscious of this divide.
</item 70>

<item 71>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
But at the same time, they're overdoing it a little bit."

Jason also chimed in on the matter, saying, "I just think the NFL is not used to celebrities coming to the games. Like basketball has it all figured out.
They're all courtside, they're sitting there. They show 'em once or twice, but then they get back to the game."

Oct. 12: Swift attends Chiefs-Broncos

"Now it's too late for you and your white horse *Bronco, to come around" - White Horse

One day after her movie premiere, Swift is in the house for the Chiefs' "Thursday Night Football" divisional matchup against the Broncos. The game is taking
place at Arrowhead Stadium, where Swift has performed herself and attended a game before. According to Al Michaels, the game will have minimal Swift coverage,
though so far there has already been a mention of the superstar.

Oct.
</item 71>

<item 72>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Taylor Swift's biggest album 1989 returns with new tracks from the vault Published 27 October

Image source, Getty Images Image caption, Taylor Swift's 1989 tour sold 2.2m tickets and earned more than $250m at the box office

On Friday, Taylor Swift released a new version of 1989 - the biggest-selling album of her career, and the one that definitively turned her into a pop star.

Featuring hits like Shake It Off, Blank Space and Style, it was originally written during the 2013-14 Red Tour, with demos stored on her phone in a folder named
"Sailor Twips".

Awarded a Grammy for album of the year, it has spent 325 weeks in the UK charts.

But now she has re-recorded it as the latest part of an ongoing campaign to regain control of her work, after an investment company bought her master tapes in
2019.

This is the biggest and riskiest part of the project.
</item 72>

<item 73>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"If she's got blue eyes, I will surmise that you'll probably date her."

The fact that all the new songs were co-written and produced by Jack Antonoff suggests there may have been rights issues over the tracks she wrote with Max
Martin - who has not been involved with the new re-recordings.

That means the vault tracks are all bathed in the dreamy, soft-focus atmospherics of Out Of The Woods, rather than the upfront pop of Shake It Off.

Speaking in 2015, Taylor admitted she'd been brutal about pruning songs from 1989's original tracklist.

"There were so many songs I wrote for this album that were really good, but if I felt in any way that they would have belonged on my last album, Red, they got
cut," she told New York radio's The Todd Show in the Morning.

"There are really, really good songs on the cutting room floor because they weren't innovative enough or new enough.
</item 73>

<item 74>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
'She dates too many men!") with a defiant "so what?", sounding much less bothered than the girl who'd written Mean as a riposte to a snarky music critic three
years earlier.

Released as 1989's first single, Taylor spent three days shooting a video, codenamed Project Sparrow, that built on the lyrical conceit.

"When I'm at award shows, I have the best time dancing [but] I'm really tall and lanky so I look a little bit like a baby giraffe learning how to walk," she
told a group of young Swifties.

"So all over the internet, it's like,'Taylor Swift looks awkward when she dances'. And as soon as I wrote [Shake It Off] I knew I wanted to do an entire music
video of me awkwardly dancing.

"I'm like, 'Yeah, I do dance awkwardly, but I'm having more fun than you.'"
</item 74>

<item 75>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Timberlake proves to be the last person you’d call in a crisis. “At some point he thought maybe music would help, so he got his guitar and he lay there with me,
strumming it,” she writes. Almost as embarrassing is husband Kevin Federline’s attempt to start a music career. “He really thought he was a rapper now,” she
writes, with the bafflement of someone who has never known pretension. “Bless his heart – because he did take it so seriously.”

Motherhood leaves her vulnerable

When Britney gets pregnant with her first son, she initially thinks it might protect her: “I wanted everyone to stay away: stand back! There’s a baby here!”
Inevitably, it doesn’t work out that way. The paparazzi become aggressive, particularly once she has her kids in quick succession – becoming pregnant a second
time three months after giving birth to her first.
</item 75>

<item 76>
When pop culture and sport collide: A timeline of Taylor Swift’s NFL takeover (The Age, 2023-09-26T05:26:05+00:00)
Unfortunately, the bracelet never made it to her wrist or his digits to her phone.

“I was disappointed that she doesn’t talk before or after her shows because she has to save her voice for the 44 songs she sings,” he said. “She doesn’t meet
anybody, or at least she didn’t want to meet me, so I took it personal[sic].” At this point, it seems like any attraction between the two was one-sided. Kelce’s
attempts to make proper contact with Swift at the show ultimately failed. Was she just not that into him? But Kelce, exhibiting the tenacity typical of a
professional sportsman, remained determined to get her attention. Wildest Beards The Kelce brothers couldn’t stop talking about Swift, even though she’d left
Travis hanging at the concert. Swift’s name began popping up everywhere on the podcast, including during conversations about... facial hair?
</item 76>

<item 77>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“This is the proudest and happiest I’ve ever felt, and the most creatively fulfilled and free I’ve ever been,” Swift told Time. “Ultimately, we can convolute it
all we want, or try to overcomplicate it, but there’s only one question... Are you not entertained?”

Swift was chosen among a strong list of contenders, including Chinese President Xi Jinping, King Charles III, the live-action Barbie movie, and Vladimir Putin.
“While her popularity has grown across the decades, this is the year that Swift, 33, achieved a kind of nuclear fusion: shooting art and commerce together to
release an energy of historic force,” the magazine said about its selection.

This year alone, Swift has broken records with her career-spanning Eras Tour, which is estimated to have grossed over $1.4bn. Her concert film, the Eras Tour
movie, has surpassed $250m at the global box office since its opening weekend in October.
</item 77>

<item 78>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
“After what we saw today, we should all be supporting Britney at this time. Regardless of our past, good and bad, and no matter how long ago it was,” Timberlake
wrote on X, formerly known as Twitter. “What’s happening to her is just not right. No woman should ever be restricted from making decisions about her own body.”

Wade Robson: 2001 to 2002

Shortly after her breakup from Timberlake, Spears was rumoured to be dating Wade Robson, a dancer and choreographer who appeared in the music video for her 2009
tune, “(You Drive Me) Crazy”. However, neither Spears nor Robson confirmed these rumours.

Colin Farrell: 2003

(Getty Images)

Back in 2003, Farrell and Spears made headlines when they were spotted holding hands at the premiere of The Recruit, sparking speculation that they were dating.
</item 78>

<item 79>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
“If the idea of my being in that place was to heal, that was not the effect.”

She’s not scared of anything after that experience, she writes, “but it doesn’t make me feel strong; it makes me sad. I shouldn’t be this strong.”

#FreeBritney protesters in 2021. Photograph: Étienne Laurent/EPA

A nurse shows Britney the blossoming fan movement that is spreading awareness of its suspicions that all is not right with her conservatorship. Come June 2021,
she calls 911 to report her father for conservatorship abuse days before a probate court hearing on the arrangement. “My voice had been used for me and against
me so many times that I was afraid nobody would recognise it now if I spoke freely,” she writes. When she is granted her freedom in November 2021, she says she
feels shock, relief, elation, sadness and joy.
</item 79>

<item 80>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
She specified that she isn’t trying to hide her relationship from the public eye.

“When you say a relationship is public, that means I’m going to see him do what he loves, we’re showing up for each other, other people are there and we don’t
care,” she said. “The opposite of that is you have to go to an extreme amount of effort to make sure no one knows that you’re seeing someone. And we’re just
proud of each other.”

During the interview with Time, the Grammy winner addressed how her appearances at Kelce’s games are heavily publicised. So far, she’s attended five of Kelce’s
games, including one on 3 December, and she’s often seated in the Kelce family suite during the events.

“There’s a camera, like, a half-mile away, and you don’t know where it is, and you have no idea when the camera is putting you in the broadcast, so I don’t know
if I’m being shown 17 times or once,” she said.
</item 80>

<item 81>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

In her forthcoming memoir, The Woman in Me, Britney Spears is expected to reflect on some of the most pivotal moments in her life, including her relationships.

Over the years, the pop star’s love life has continued to make headlines. Most recently, her ex-husband Sam Asghari filed for divorce after only 14 months of
marriage.
</item 81>

<item 82>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She has never really known freedom

As a child, Britney escaped her difficult home life – her alcoholic father often rowing with her mother – into the outdoors, which “gave me a sense of aliveness
and danger”, she writes. Performing also makes her feel invincible. But by the age of 16, she is public property – incapable of going outside without being
mobbed, and her love of singing and dancing now a lucrative resource.

By the time of her second album, she wants for so little: skinny-dipping with her dancers after playing Rock in Rio in January 2001 is one of her happiest
memories. “It was a taste of rebellion and freedom, but I was just having fun and being a 19-year-old,” she writes.
</item 82>

<item 83>
Looking for a book to read? Here are 14 tips for October (The Sydney Morning Herald, 2023-10-01T21:04:20+00:00)
No doubt the updated edition will have the verdict on SBF. The Woman in Me

Britney Spears, Simon & Schuster, $49.99. October 25 The huge-selling American singer has negotiated the slings and arrows of outrageous fortune in her 41
years, but at the heart of this book – said to have earned her a $US15 million ($23.3 million) advance – is the controversial “conservatorship” by which her
father and others were given almost complete control of her life, but still failed to stop it going off the rails many times.

Sure to be a bestseller, the book is described as a candid account of her life – exactly what her fans want. Mr Einstein’s Secretary

Matthew Reilly, Macmillan, $44.99. October 17 Hannah Fischer lived her early life in Berlin, and her next door neighbour was Albert Einstein – he seems to be
having a popular culture moment. She idolised him, and she loved physics.
</item 83>

<item 84>
From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
They made their relationship Instagram official in 2017 and went on to support each other through different milestones - including the 29th annual GLAAD Media
Awards, where Spears accepted the Vanguard Award for her work promoting equality and acceptance.

In April 2022, Spears announced she was pregnant with her and Asghari’s first child together, and her third child. However, the following month, she said in a
statement posted to Instagram that she had suffered a miscarriage. The couple wrote: “We have lost our miracle baby.”

That June, they held their lavish nuptials at her home in California with a star-studded guest list, which included Selena Gomez, Paris Hilton, Drew Barrymore,
and Donatella Versace - who designed Spears’ wedding dress.

In August, months after Asghari shut down speculation about getting a divorce, TMZ claimed that he had confronted Spears about rumours she cheated on him.
</item 84>

<item 85>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
The conservatorship stripped me of my womanhood, made me into a child.” It’s hard to explain, she says, “how quickly I could vacillate between being a little
girl and being a teenager and being a woman, because of the way they had robbed me of my freedom … They wanted me to be wild onstage, the way they told me to
be, and to be a robot the rest of the time.”

Music seems to have become an afterthought

Early on, Britney writes, she loved making music. She told the producer Max Martin that she wanted to sound more R&B than straight pop; she listened to Tainted
Love the night before recording Baby, One More Time and stayed up late so that she would sound as gravelly and “fried” as Marc Almond. She was a studio
perfectionist who would stay in the booth for hours, and the school setting and uniforms of the Baby video were her idea.
</item 85>

<item 86>
Taylor Swift-Travis Kelce timeline: Everything to know about rumored romance between pop star, Chiefs TE (CBSSports.com, 2023-10-12T23:21:52+00:00)
How do you not get that? That's called images right there."

Oct. 1: Swift attends Chiefs-Jets

"Welcome to New York, it's been waiting for you," - Welcome to New York

OK, so the Jets technically play in New Jersey, but we're just basing this off their name. Swift went to her second consecutive Chiefs game, showing up with
some famous friends, Blake Lively, Ryan Reynolds and Hugh Jackman. In the week leading up to the game, there were reports that the singer would be at the game.
The Chiefs defeated the Jets 23-20.

Oct. 5: NFL defends itself

Swift coverage is everywhere and the league even changed their social media bio to one of the singer's lyrics. The league's X (formerly Twitter) bio to read,
"NFL (Taylor's Version)," while their Instagram account read, "Chiefs are 2-0 as Swifties."
</item 86>

<item 87>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
No-one, it turns out.

The song is thematically similar to Blank Space, commenting on the media's portrayal of Taylor's relationships: "But if I'm all dressed up / They might as well
be lookin' at us / And if they call me a slut / You know it might be worth it for once."

The best of the new tracks is Is It Over Now. An angrier take on 1989's philosophical break-up songs, it calls back to the Harry Styles snowmobile incident (see
also, Out Of The Woods) then, deliciously, references the "Sad Taylor Boat" meme - a photo of Taylor fleeing a vacation alone after the couple split up.

She goes on to call an ex a "lying traitor", accuses them of parading their new relationships in public, and noting that every new girlfriend looks like her.

"If she's got blue eyes, I will surmise that you'll probably date her."
</item 87>

<item 88>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
"Whereas now I go in thinking, 'How long do we have on the clock before something comes along and puts a wrench in it, or your publicist calls and says this
isn't a good idea?"

Interestingly, an early draft of the lyrics had the couple fleeing the city and blotting out the pressures of the world, an idea which eventually resurfaced in
I Know Places.

10) How You Get The Girl

What's that? An acoustic guitar? Surely not...

This is one of the few 1989 tracks you can imagine working as a country song, if it was stripped of its whip-crack snare drums and buzzing basslines.

Over that backing, Taylor turns agony aunt, dispensing advice on how to win back someone's heart... although her solutions (standing outside their house in the
rain, bombarding them with cute photos) suggest she hasn't quite weaned herself off Love, Actually just yet.

"It's kind of a tutorial.
</item 88>

<item 89>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

It may be over a decade since Britney Spears and Kevin Federline ended their marriage, but the former couple continue to have their ups and downs when it comes
to co-parenting their two teenage sons, Sean Preston and Jayden.

Ahead of the release of the pop star’s memoir, The Woman in Me, which is due for release on 24 October, fans are anticipating Spears will share new details
about her relationship with her second husband, Federline.
</item 89>

<item 90>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
“Whether he realises it or not, Mr Federline has not only violated the privacy and dignity of the mother of his children, he has undermined his own children,
whose privacy he should protect,” the statement read. “Putting aside his ITV interview, Mr Federline’s ill-advised decision to post an old video of his 11 and
12-year-old children was cruel, bottom of the barrel stuff. It was abhorrent.”

“In addition to demeaning himself and violating societal norms, he has now also created various legal issues for himself, including, but not limited to,
implicating cyber-harassment and cyber-bullying statutes, among other things.”

The Woman in Me hits shelves in the US and UK on 24 October.
</item 90>

<item 91>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Justin had embarrassed me publicly, so my rebuttal onstage was to kind of go there a little bit, too. But it was absolutely horrible.”

She has an immaculate sense of shade

Britney seems too pure of heart for active malice, but she has an immaculate way with a piercing characterisation. Timberlake’s band, ‘NSync, were big hip-hop
heads, she writes, and sometimes “tried too hard to fit in” with Black artists. When Timberlake spots Ginuwine at an event, he “got all excited and said, so
loud, ‘Oh yeah fo shiz, fo shiz! Ginuwiiiine! What’s up, homie!’”

Her account of being encouraged to abort their pregnancy at home, lest anyone see them going in and out of hospital, is horrifying. Britney, with no pain
relief, is writhing on the floor. Timberlake proves to be the last person you’d call in a crisis.
</item 91>

<item 92>
Britney Spears and Kevin Federline: A complete timeline of their relationship (The Independent - Life and Style, 2023-10-20T15:02:26+00:00)
After months of court discussions, Federline rejected Spears’s child support proposal and instead doubled his request to $40k a month. A settlement was finally
reached between the two parties in September 2018, with Spears increasing her child support payments to Federline from $20,000 to $35,000 a month.

Amid her father’s health crisis and ongoing emotional distress, Spears voluntarily checked into a mental health treatment facility in 2019. After hearing the
news, Federline’s attorney Mark Kaplan told Entertainment Tonight: “Kevin commends Britney for recognising that she needs to take a step back and that she is
taking the reasonable steps to dealing with her situation in a responsible way. He wishes everyone well.” That same year, Federline was granted 70 percent
custodial rights and Spears 30 per cent.

After years with her father in control of her finances and personal life, Spears was finally granted freedom from her conservatorship in November 2021.
</item 92>

<item 93>
Taylor Swift's 1989: The stories behind her biggest album (BBC News - Entertainment & Arts, 2023-10-13T23:00:08+00:00)
Those search engine clicks gotta be lucrative, right?

Image source, Getty Images Image caption, Many of 1989's songs were supposedly inspired by Taylor's relationship with Harry Styles

3) Style

An epic aural painting of windswept coastlines and propulsive wanderlust, Style is (supposedly) named for Harry Styles, with whom Taylor had a brief
relationship in 2012.

Except she says it was inspired by fashion, and how certain items like a black dress or red lipstick will "never go out of style". That prompted her to write
about the people who'll always be part of your life, even after the flame of romantic love wanes (all of which sounds like a classic piece of deflection to a
probing journalist).

Whichever interpretation you prefer, the song marks a shift in Swift's psychology, as she briefly admits to having been unfaithful.

"I would never have said anything like that on a previous album," she told Ryan Seacrest.
</item 93>

<item 94>
Travis Kelce admits the NFL is overdoing it with their Taylor Swift coverage (The Independent - Life and Style, 2023-10-04T19:58:00+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

Travis Kelce believes the National Football League’s coverage of his rumoured girlfriend Taylor Swift is a bit too much.

Following his game against the New York Jets on 1 October, the 33-year-old Kansas City Chiefs tight end opened up about the NFL broadcast on the most recent
episode of his podcast, New Heights with Jason and Travis Kelce.
</item 94>

<item 95>
Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' (FOX News - Lifestyle, 2023-11-11T10:00:36+00:00)
But it was the comment from the A-list singer that really got the girls' attention.

Swift liked and commented on the viral TikTok saying, "Wait, this is amazing."

"That was the best surprise ever — my year is made," Henslee told Fox News Digital in reference to Swift's comment.

TAYLOR SWIFT QUIZ! HOW WELL DO YOU KNOW THE FAMOUS POP MUSICIAN?

The girls usually can be found listening to Swift while making her popular friendship bracelets or playing soccer with one another.

"Her like/comment is something the girls and this Swiftie mama will always remember," Tia commented.

This isn't the first time a young person went viral for trying to open a CD.

In May, Fox News Digital reported on a mom who showed her 10-year-old daughter's struggle as she attempted to pop a "Now That's What I Call Music! 21" CD
(2006).

CDs were first introduced in 1982, according to the tech company Philips' website.
</item 95>

<item 96>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
“Over the years, I’ve learned I don’t have the time or bandwidth to get pressed about things that don’t matter,” Swift told Time. “Yes, if I go out to dinner,
there’s going to be a whole chaotic situation outside the restaurant. But I still want to go to dinner with my friends.

“Life is short. Have adventures. Me locking myself away in my house for a lot of years - I’ll never get that time back,” she added. “I’m more trusting now than
I was six years ago.”
</item 96>

<item 97>
Taylor Swift is Time’s Person of the Year (The Independent - Life and Style, 2023-12-06T13:55:17+00:00)
The decision came after one of the singer’s fans, 23-year-old Ana Clara Benevides, died from a cardiac arrest during Swift’s first performance in the country.
Swift explained to Time that “part of my identity as a human” is putting on a show for her fans, no matter the circumstances. “I know I’m going on that stage
whether I’m sick, injured, heartbroken, uncomfortable, or stressed,” she said. “That’s part of my identity as a human being now. If someone buys a ticket to my
show, I’m going to play it unless we have some sort of force majeure.”

Perhaps it was her decision to re-record her music catalogue that catapulted Swift into such success. When music mogul Scooter Braun aquired Swift’s longtime
label, Big Machine Records, in 2019, he also gained the rights to the master recordings of Swift’s first six studio albums. In November 2020, Braun sold the
masters to an investment fund in a deal that was believed to be over $300 million.
</item 97>

<item 98>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Her life under the conservatorship is a horror story

It feels fitting that The Woman in Me is being released a week before Halloween. No matter how much we already know about the 13 years that Britney lived under
a conservatorship, her recounting of its imposition, life under it and her inability to escape it is nothing short of a horror story. It is appalling, like
something out of a Victorian novel, not the very real experiences of a thirtysomething in the last decade.

When she fears that Federline is going to take away access to her kids, she locks herself in a bathroom with one of her sons. “Before I knew what was happening,
a Swat team in black suits burst through the bathroom door as if I’d hurt someone,” she writes, and she is subject to a forced hospital hold. Soon after, her
mother invites her to her beach house because “the cops are after you”. When she arrives, so does another Swat team.
</item 98>

<item 99>
Taylor Swift reveals secret start to her relationship with Travis Kelce (The Independent - Life and Style, 2023-12-06T14:23:01+00:00)
In an interview withWSJ Magazine, published on 20 November, he described how much he admires her skills as a musician. “I’ve never been a man of words. Being
around her, seeing how smart Taylor is, has been f***ing mind-blowing. I’m learning every day,” he explained.

However, he also acknowledged the heavy public attention around his private life as a result of his relationship, noting he’s “never dated anyone with that kind
of aura around them”. But, according to Kelce, he’s not letting the media affect his relationship.

“But at the same time, I’m not running away from any of it,” he said. “The scrutiny she gets, how much she has a magnifying glass on her, every single day,
paparazzi outside her house, outside every restaurant she goes to, after every flight she gets off, and she’s just living, enjoying life. When she acts like
that I better not be the one acting all strange.”
</item 99>

<item 100>
Young Taylor Swift fans go viral for struggling to open the star's CD: 'Making me laugh' (FOX News - Lifestyle, 2023-11-11T10:00:36+00:00)
She actually saw me and liked it. It’s crazy."

THE TAYLOR SWIFT OBSESSION: PSYCHOLOGIST WEIGHS IN ON WHY FANS WORSHIP CELEBRITIES

In a second video, the girls were captured using the CD player, which proved to be slightly easier to figure out.

All three "Swifties" can be heard "oohing" and "aahing" as they learn how to open the disc player.

"Whoa! That was fancy, guys. This is a fancy one," one of the girls can be heard saying in the TikTok video.

TAYLOR SWIFT'S EXES: BABY NAMES INSPIRED BY THE SINGER'S FORMER FLAMES REVEALED

The Gen Zers captured the attention of viewers online who could not help but comment on the viral moment — and the overwhelming response was "I feel old."

But it was the comment from the A-list singer that really got the girls' attention.
</item 100>
=== END ITEM id=e30baef8283a3419 ===

=== ITEM id=dcc0a902070a6782 ===
Gold sentences and facts:
Sentence 0: “I’m here until the end,” she said in her press conference on Friday.
  Fact 0.0: A female speaker said 'I'm here until the end' in her press conference on Friday.
Sentence 1: He has been out of work since his Blues tenure came to a close back in April, but it would be a hard sell to fans after his time with the London club went so poorly.
  Fact 1.0: A man has been out of work since his tenure with the Blues came to a close back in April.
  Fact 1.1: It would be a hard sell to fans for this man to return, since his time with the London club went so poorly.

Retrieved items:

=== END ITEM id=dcc0a902070a6782 ===
